import numpy as np
import os
import random
import torch
import tqdm

from trainer_base import BaseTrainer
from utils import get_sep_position, batch_ids


class AuxiliarMasksRemovalTrainer(BaseTrainer):
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)

        # train to produce all masking rates of COTs at once (uniformly sample p per each sample) 
        self.joint_masked_distribution = args.joint_masked_distribution

        # schedule specific p per each epoch, p = rate of tokens to mask in COT
        self.remove_by_schedule = args.remove_by_schedule

        # list of (from_epoch, p)
        self.masks_removal_schedule = args.removal_schedule
        self.schedule_index = 0

        # p% of tokens are masked from left to right
        self.left_to_right_removal = args.left_to_right_removal
        # p% of tokens are masked contiguously starting from random pos
        self.random_contiguous_removal = args.random_contiguous_removal

        self.removal_p = self.masks_removal_schedule[self.schedule_index][1]
        self.no_cot_stage = self.args.no_cot_stage

        if self.no_cot_stage:
            self.val_removal_ps = [0.0, 1.0]
        else:
            self.val_removal_ps = [1.0, 0.5, 0.75, 0.9, 0.95, 1.0]

        self.val_truncation_kwargs = {
            "eval_flag": self.joint_masked_distribution
        }
        self.val_generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "stop_on_two_eos": True,
            "use_inputs_cot": False,
            "position_ids_shift": self.args.keep_position and self.joint_masked_distribution and self.left_to_right_removal,
            "insert_const_ids_in_cot": False,
            "random_insertion_prob": None
        }

        self.generative_eval_hooks = []
        if self.val_generation_kwargs["position_ids_shift"]:
            self.generative_eval_hooks.append(self.generation_eval_position_ids_shift_hook)
        
        # For generative eval in case of left_to_right_removal & joint_masked_distribution
        self.n_tokens_removed = None

        # For random masking 
        self.mask_id = torch.tensor(self.tokenizer.encode("Mask"))
        if len(self.mask_id) != 1:
            self.mask_id = torch.tensor(self.tokenizer.encode("M"))
            if len(self.mask_id) != 1: # supposing the tokenizer prepends BOS
                self.mask_id = self.mask_id[-1]
        self.mask_id = self.mask_id.to(self.device)

        # For eval with mask insertion
        self.insert_const_ids_func = None
        if self.args.replace_mask:
            self.val_generation_kwargs["insert_const_ids_in_cot"] = True
            self.insert_const_ids_func = self.sample_random_contiguous_mask
            self.generative_eval_hooks.append(self.generation_eval_insert_ids_hook)

            if not (self.left_to_right_removal or self.random_contiguous_removal):
                self.insert_const_ids_func = self.mask_token_insert_func

        # Metrics
        additional_metrics = ["removal_p"]
        if self.jepa_training:
            additional_metrics.append("CE_loss")
            additional_metrics.append("JEPA_loss")
        self.setup_metrics(additional_metrics=additional_metrics)
    
    def _train_process(self):
        step = 0

        if self.args.from_pretrained_checkpoint:
            step = self._resume_checkpoint(self.args.from_pretrained_checkpoint)

        # If JEPA model starts training from plain full-COT model
        if self.args.from_pretrained_fullcot_checkpoint:
            step = self._resume_checkpoint(self.args.from_pretrained_fullcot_checkpoint, load_optimizer_state=False)

        if self.accelerator.is_main_process and self.writer:
            self.writer.set_step(step, mode="train")

        all_cot_removed_in_batch = False
        loss_log = []
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="train")
                self.writer.add_scalar("epoch", epoch)
            self.model.train()

            if self.remove_by_schedule and self.schedule_index + 1 < len(self.masks_removal_schedule):
                if self.masks_removal_schedule[self.schedule_index + 1][0] == epoch:
                    self.schedule_index += 1
                    self.removal_p = self.masks_removal_schedule[self.schedule_index][1]
                    if self.args.reset_optimizer and (not all_cot_removed_in_batch):
                        self._reset_optimizer()

            loss_log.append([])

            print_line = f"Epoch {epoch}; Step {step}"
            if not self.joint_masked_distribution:
                print_line += f"; Scheduled to remove: {self.removal_p * 100}% of COT"
            print(print_line)
            
            if self.accelerator.is_main_process:
                print(f">>> Using mixed precision: {self.accelerator.state.mixed_precision}")

            for batch in tqdm.tqdm(self.train_dataloader):
                batch, all_cot_removed_in_batch = self.process_input_truncation(batch)
                # self.move_batch_to_device(batch, self.device) <-- handled by accelerator

                # print(batch["input_ids"].shape, batch["input_ids"][0])
                # print(self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)[0])

                # if not all_cot_removed_in_batch:
                #     best_val_accuracy = float('-inf')
                if self.args.max_len_train > 0 and batch["input_ids"].shape[-1] > self.args.max_len_train:
                    print ('skipped')
                    continue

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        outputs = self.model.compute_loss(**batch)
                        loss = outputs.loss
                        logits = outputs.logits
                        
                    loss_log[-1].append(loss.item())
                    self.accelerator.backward(loss.div(self.args.accumulate))
                    grad_norm = self.get_grad_norm()

                    if self.jepa_training:
                        self.model.update_ref_model()

                    torch.nn.utils.clip_grad_norm_(self.get_trainable_params(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.metrics_tracker is not None:
                    self.metrics_tracker.update("loss", loss.item())
                    self.metrics_tracker.update("perplexity", loss.exp().item())
                    self.metrics_tracker.update("token_accuracy", outputs.token_accuracy.item())
                    self.metrics_tracker.update("grad_norm", grad_norm)
                    self.metrics_tracker.update("removal_p", self.removal_p)

                    if self.jepa_training:
                        self.metrics_tracker.update("CE_loss", outputs.ce_loss.item())
                        self.metrics_tracker.update("JEPA_loss", outputs.logits_loss.item())
                
                if self.accelerator.is_main_process and step % 100 == 0:
                    token_accuracy = outputs.token_accuracy.item()
                    ppl = loss.exp().item()
                    print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                    if self.metrics_tracker is not None:
                        self.writer.set_step(step, mode="train")
                        self.log_scalars(self.metrics_tracker)
                        self.metrics_tracker.reset()
                        self.writer.add_scalar("learning rate", self.optimizer.param_groups[0]['lr'])

                step += 1

            loss_log[-1] = sum(loss_log[-1]) / len(loss_log[-1])

            self.evaluate(step)

            self.save_epoch(epoch)
    
    def evaluate(self, step):
        for val_removal_p in self.val_removal_ps:
            print(f"\nVALIDATION ON VAL_REMOVAL_P = {val_removal_p}\n")
            name = f"val_{val_removal_p}" if val_removal_p != 1.0 else "val"
                        
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode=name)
            
            if val_removal_p == 1.0 and self.args.replace_mask:
                self.val_generation_kwargs["use_inputs_cot"] = True
            else:
                self.val_generation_kwargs["use_inputs_cot"] = False

            if not (self.left_to_right_removal or self.random_contiguous_removal):
                self.val_generation_kwargs["random_insertion_prob"] = val_removal_p if val_removal_p > 0.0 else None

            self._evaluate(
                dataloader=self.val_dataloader,
                name=name,
                truncation_kwargs={"val_removal_p": val_removal_p},
                generation_kwargs=self.val_generation_kwargs,
                perform_generative_eval=True,
                generative_eval_hooks=self.generative_eval_hooks if val_removal_p > 0.0 else [],
                generative_eval_single_batch_size=self.val_generation_kwargs["insert_const_ids_in_cot"]
            )
    
    def generation_eval_insert_ids_hook(self, batch, truncation_kwargs, generation_kwargs):
        # Regenerate with no trunc
        if generation_kwargs.get("insert_const_ids_in_cot", False):
            ids_to_insert, insert_position = self.insert_const_ids_func(batch, **truncation_kwargs)
            generation_kwargs["ids_to_insert"] = ids_to_insert
            generation_kwargs["insert_position"] = insert_position
        return generation_kwargs
    
    def generation_eval_position_ids_shift_hook(self, generation_kwargs, *args, **kwargs):
        if generation_kwargs.get("position_ids_shift", None) is not None:
            generation_kwargs["position_ids_shift"] = getattr(self, "n_tokens_removed", None)
        return generation_kwargs

    def mask_token_insert_func(self, batch, val_removal_p):
        return self.mask_id.clone().detach().unsqueeze(0), None

    def sample_random_contiguous_mask(self, batch, val_removal_p):
        input_ids = batch["input_ids"]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)

        assert self.same_elements(first_sep_positions) \
            and self.same_elements(second_sep_positions) \
            and self.same_elements(eos_positions)
    
        length = second_sep_positions[0] - first_sep_positions[0] - 1
        n_tokens_to_remove = int(np.round(val_removal_p * length.item()))

        insert_position = 0 if self.left_to_right_removal else random.choice(torch.arange(0, length - n_tokens_to_remove + 1))
        ids_to_insert = torch.full((input_ids.shape[0], n_tokens_to_remove), self.mask_id).to(self.device)

        return ids_to_insert, insert_position
        
    def process_input_truncation(self, batch, val_removal_p=None):
        input_ids = batch['input_ids']
        labels = batch['labels']

        if val_removal_p is not None:
            if self.left_to_right_removal or self.random_contiguous_removal:
                return self.contiguous_truncation(
                    input_ids=input_ids,
                    labels=labels,
                    removal_p=val_removal_p,
                    joint_masked_distrubution=False,
                    left_to_right=self.left_to_right_removal,
                    compute_full_input_ids=self.jepa_training
                )
            return self.random_masking(
                input_ids=input_ids,
                labels=labels,
                removal_p=val_removal_p,
                joint_masked_distrubution=False,
                compute_full_input_ids=self.jepa_training
            )

        if self.removal_p == 0.0 and (not self.joint_masked_distribution):
            batch = {
                "input_ids": input_ids,
                "labels": labels,
                "full_input_ids": input_ids.clone(),
                "full_labels": labels.clone(),
                "position_ids": None
            }
            return batch, False

        if self.left_to_right_removal or self.random_contiguous_removal:
            return self.contiguous_truncation(
                input_ids=input_ids,
                labels=labels,
                removal_p=self.removal_p,
                joint_masked_distrubution=self.joint_masked_distribution,
                left_to_right=self.left_to_right_removal,
                compute_full_input_ids=self.jepa_training
            )
        return self.random_masking(
            input_ids=input_ids,
            labels=labels,
            removal_p=self.removal_p,
            joint_masked_distrubution=self.joint_masked_distribution,
            compute_full_input_ids=self.jepa_training
        )

    def _get_position_ids(self, input_ids):
        if self.args.keep_position:
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.device)
            if input_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(0).repeat(input_ids.shape[0], 1)
            return position_ids
        return None
    
    def _get_sep_positions(self, input_ids):
        first_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)
        return first_sep_positions, second_sep_positions, eos_positions

    @staticmethod
    def same_elements(x):
        return (x == x[0]).all()

    def _get_prefix_contiguous_truncation(self, n_tokens_to_remove, left_to_right=True, L=0):
        if self.no_cot_stage:
            prefix = torch.full((1,), self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
            ignored_prefix_labels = torch.full_like(prefix, -100)
            return prefix, ignored_prefix_labels
        
        if left_to_right:
            prefix_str = f" ## masked {n_tokens_to_remove} ## "
        else:
            prefix_str = f" ## masked {n_tokens_to_remove} from {L} ## "

        prefix = self.tokenizer(
            prefix_str,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].to(self.device).squeeze(0)

        eos = torch.full((1,), self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
        prefix = torch.cat([prefix, eos])

        ignored_prefix_labels = torch.full_like(prefix, -100)
        return prefix, ignored_prefix_labels

    def _concat_ids(self, ids, prefix_ids, cot_start, start, end, eos, dim, labels_flag=False):
        ids_to_cat = [ids[..., :cot_start - 1], prefix_ids] # move EOS_TOKEN_ID in prefix
        if cot_start < start:
            ids_to_cat.append(ids[..., cot_start:start])

        if self.args.replace_mask:
            if labels_flag and (self.args.replace_mask_in_labels is False):
                # Recover all COT tokens in labels in case of mask replacement
                ids_to_cat.append(ids[..., start:end])
            else:
                mask_shape = list(ids.shape)
                mask_shape[-1] = (end - start).item()
                mask_tensor = torch.full(size=mask_shape, fill_value=self.mask_id.item(), dtype=torch.long, device=ids.device)
                ids_to_cat.append(mask_tensor)
        
        ids_to_cat.append(ids[..., end:eos + 1])

        ids_new = torch.cat(ids_to_cat, dim=dim)
        return ids_new

    def _stack_full_inputs(self, full_input_ids_new, compute_full_input_ids):
        if not compute_full_input_ids:
            return None, None, None

        full_labels_new = full_input_ids_new.clone()
        full_position_ids_new = self._get_position_ids(full_input_ids_new)

        return full_input_ids_new, full_labels_new, full_position_ids_new
    
    def _contiguous_truncation(
            self,
            input_ids,
            labels,
            removal_p,
            first_sep_positions,
            eos_positions,
            nonmasked_lengths,
            batched=False,
            left_to_right=True,
            compute_full_input_ids=False
    ):
        length = nonmasked_lengths[0] if batched else nonmasked_lengths
        n_tokens_to_remove = int(np.round(removal_p * length.item()))

        cot_start = first_sep_positions[0] + 1 if batched else first_sep_positions + 1
        offset = 0 if left_to_right else random.choice(torch.arange(0, length - n_tokens_to_remove + 1))

        start = cot_start + offset
        end = start + n_tokens_to_remove
        eos = eos_positions[0] if batched else eos_positions

        prefix, ignored_prefix_labels = self._get_prefix_contiguous_truncation(n_tokens_to_remove, left_to_right, offset)

        if input_ids.ndim == 2:
            batch_size = input_ids.shape[0]
            prefix, ignored_prefix_labels = prefix.unsqueeze(0), ignored_prefix_labels.unsqueeze(0)
            prefix, ignored_prefix_labels = prefix.repeat(batch_size, 1), ignored_prefix_labels.repeat(batch_size, 1)

        input_ids_new = self._concat_ids(input_ids, prefix, cot_start, start, end, eos, dim=int(batched))
        labels_new = self._concat_ids(labels, ignored_prefix_labels, cot_start, start, end, eos, dim=int(batched), labels_flag=True)

        full_input_ids_new = self._concat_ids(
            input_ids, prefix, cot_start, cot_start, cot_start, eos, dim=int(batched)
        ) if compute_full_input_ids else None
        full_input_ids_new, full_labels_new, full_position_ids_new = self._stack_full_inputs(full_input_ids_new, compute_full_input_ids)
        
        if self.args.keep_position:
            length = max(input_ids.shape[-1], input_ids_new.shape[-1])

            position_ids_new = torch.arange(
                0, length, dtype=torch.long, device=self.device
            )
            if input_ids.ndim == 2:
                position_ids_new = position_ids_new.unsqueeze(0).repeat(batch_size, 1)
            
            position_ids_new[..., start + len(prefix) - 1:] += end - start
            position_ids_new = position_ids_new[..., :input_ids_new.shape[-1]]
        else:
            position_ids_new = None
        
        nonmasked_lengths = nonmasked_lengths - (end - start)

        if batched:
            n_tokens_to_remove = torch.full((batch_size,), n_tokens_to_remove, dtype=torch.long, device=self.device)

        batch = {
            "input_ids": input_ids_new,
            "labels": labels_new,
            "position_ids": position_ids_new,
            "full_input_ids": full_input_ids_new,
            "full_labels": full_labels_new,
            "full_position_ids": full_position_ids_new,
            "nonmasked_lengths": nonmasked_lengths,
            "n_tokens_removed": n_tokens_to_remove
        }

        return batch

    def contiguous_truncation(
            self,
            input_ids,
            labels,
            removal_p,
            joint_masked_distrubution,
            left_to_right=True,
            compute_full_input_ids=False
    ):
        batch_size = input_ids.shape[0]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)
        nonmasked_lengths = second_sep_positions - first_sep_positions - 1
        
        all_cot_removed_in_batch = True
        position_ids_new = None

        same_cots_flag = \
            self.same_elements(nonmasked_lengths) \
            and self.same_elements(first_sep_positions) \
            and left_to_right \
            and not joint_masked_distrubution

        if same_cots_flag:
            # ALL COT HAVE THE SAME SIZE AND POSITION
            batch = self._contiguous_truncation(
                input_ids,
                labels,
                removal_p,
                first_sep_positions,
                eos_positions,
                nonmasked_lengths,
                batched=True,
                left_to_right=left_to_right,
                compute_full_input_ids=compute_full_input_ids
            )
            # For generative eval
            self.n_tokens_removed = batch["n_tokens_removed"]
            all_cot_removed_in_batch = not bool(batch["nonmasked_lengths"].sum())
            return batch, all_cot_removed_in_batch
        
        input_ids_new = []
        labels_new = []
        full_input_ids_new = [] if compute_full_input_ids else None
        position_ids_new = [] if self.args.keep_position else None
        n_tokens_removed = []

        for batch_idx in range(batch_size):
            if joint_masked_distrubution:
                removal_p = random.uniform(0, 1)

            batch_i = self._contiguous_truncation(
                input_ids[batch_idx],
                labels[batch_idx],
                removal_p,
                first_sep_positions[batch_idx],
                eos_positions[batch_idx],
                nonmasked_lengths[batch_idx],
                batched=False,
                left_to_right=left_to_right,
                compute_full_input_ids=compute_full_input_ids
            )

            input_ids_new.append(batch_i["input_ids"])
            labels_new.append(batch_i["labels"])
            if compute_full_input_ids:
                full_input_ids_new.append(batch_i["full_input_ids"])
            if self.args.keep_position:
                position_ids_new.append(batch_i["position_ids"])
            nonmasked_lengths[batch_idx] = batch_i["nonmasked_lengths"]
            n_tokens_removed.append(batch_i["n_tokens_removed"])
        
        input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
        if self.args.keep_position:
            position_ids_new = batch_ids(position_ids_new, 0, self.device, torch.long)

        full_input_ids_new = batch_ids(
            full_input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype
        ) if compute_full_input_ids else None
        full_input_ids_new, full_labels_new, full_position_ids_new = self._stack_full_inputs(full_input_ids_new, compute_full_input_ids)
        
        batch = {
            "input_ids": input_ids_new,
            "labels": labels_new,
            "position_ids": position_ids_new,
            "full_input_ids": full_input_ids_new,
            "full_labels": full_labels_new,
            "full_position_ids": full_position_ids_new
        }

        # For generative eval
        self.n_tokens_removed = torch.tensor(n_tokens_removed, dtype=torch.long, device=self.device)
        all_cot_removed_in_batch = not bool(nonmasked_lengths.sum())
        
        return batch, all_cot_removed_in_batch

    @staticmethod
    def _get_random_cot_mask(cot_start, cot_end, n_tokens_to_remove):
        removed_indices = cot_start + torch.randperm(
            cot_end - cot_start,
            device=cot_start.device,
            dtype=torch.long
        )[:n_tokens_to_remove]
        removed_indices, _ = torch.sort(removed_indices)

        all_indices = torch.arange(cot_start, cot_end, device=cot_start.device)
        mask = torch.isin(all_indices, removed_indices)
        remaining_indices = all_indices[~mask]
        return remaining_indices, removed_indices, mask

    def _get_prefix_random_masking(self, cot_start, removed_indices):
        if self.no_cot_stage:
            prefix = torch.full((1,), self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
            ignored_prefix_labels = torch.full_like(prefix, -100)
            return prefix, ignored_prefix_labels
        
        if self.args.replace_mask:
            prefix_str = f" ## masked {len(removed_indices)} ## "
        else:
            prefix_str = f" ## removed tokens: {(removed_indices - cot_start).tolist()} ## "

        prefix = self.tokenizer(
            prefix_str,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].to(self.device).squeeze(0)

        eos = torch.full((1,), self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
        prefix = torch.cat([prefix, eos])

        ignored_prefix_labels = torch.full_like(prefix, -100)
        return prefix, ignored_prefix_labels        

    def random_masking(
            self,
            input_ids,
            labels,
            removal_p,
            joint_masked_distrubution,
            compute_full_input_ids=False
    ):
        batch_size = input_ids.shape[0]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)
        nonmasked_lengths = second_sep_positions - first_sep_positions - 1

        all_cot_removed_in_batch = True

        input_ids_new = []
        labels_new = []
        full_input_ids_new = [] if compute_full_input_ids else None
        position_ids_new = [] if self.args.keep_position else None

        for batch_idx in range(batch_size):
            if joint_masked_distrubution:
                removal_p = random.uniform(0, 1)
            n_tokens_to_remove = int(np.round(removal_p * nonmasked_lengths[batch_idx].item()))

            cot_start = first_sep_positions[batch_idx] + 1
            cot_end = second_sep_positions[batch_idx]
            eos = eos_positions[batch_idx]

            remaining_indices, removed_indices, mask = self._get_random_cot_mask(cot_start, cot_end, n_tokens_to_remove)
            prefix, ignored_prefix_labels = self._get_prefix_random_masking(cot_start, removed_indices)

            if not self.args.replace_mask: # truncate selected tokens
                if remaining_indices.shape[-1]:
                    input_ids_cot_masked = input_ids[batch_idx, remaining_indices].detach().clone()
                    input_ids_insert = torch.cat([prefix, input_ids_cot_masked])
                    labels_insert = torch.cat([ignored_prefix_labels, input_ids_cot_masked.clone()])
                else:
                    input_ids_insert = prefix
                    labels_insert = ignored_prefix_labels
            else:
                input_ids_cot_masked = input_ids[batch_idx, cot_start:cot_end].detach().clone()
                input_ids_cot_masked[mask] = self.mask_id
                input_ids_insert = torch.cat([prefix, input_ids_cot_masked])

                if self.args.replace_mask_in_labels:
                    labels_insert = torch.cat([ignored_prefix_labels, input_ids_cot_masked.clone()])
                else:
                    input_ids_cot = input_ids[batch_idx, cot_start:cot_end].detach().clone()
                    labels_insert = torch.cat([ignored_prefix_labels, input_ids_cot])

            input_ids_new.append(torch.cat([
                input_ids[batch_idx, :cot_start - 1],  # move EOS_TOKEN_ID in prefix
                input_ids_insert,
                input_ids[batch_idx, cot_end:eos + 1]
            ]))
            labels_new.append(torch.cat([
                labels[batch_idx, :cot_start - 1],  # move EOS_TOKEN_ID in prefix
                labels_insert,
                labels[batch_idx, cot_end:eos + 1]
            ]))

            if compute_full_input_ids:
                full_input_ids_new.append(torch.cat([
                    input_ids[batch_idx, :cot_start - 1],  # move EOS_TOKEN_ID in prefix
                    prefix,
                    input_ids[batch_idx, cot_start:eos + 1].detach().clone()
                ]))

            if self.args.keep_position:
                length = max(input_ids[batch_idx].shape[-1], input_ids_new[-1].shape[-1])
                position_ids_new.append(torch.arange(
                    0, length, dtype=torch.long, device=self.device
                ))

                mask = torch.zeros_like(input_ids[batch_idx], dtype=torch.long)
                mask[removed_indices] = 1
                mask_cumsum = torch.cumsum(mask)
                position_ids_new[-1] +=  mask_cumsum

            nonmasked_lengths[batch_idx] = nonmasked_lengths[batch_idx] - n_tokens_to_remove

        input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)

        full_input_ids_new = batch_ids(
            full_input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype
        ) if compute_full_input_ids else None
        full_input_ids_new, full_labels_new, full_position_ids_new = self._stack_full_inputs(full_input_ids_new, compute_full_input_ids)
        
        if self.args.keep_position:
            position_ids_new = batch_ids(position_ids_new, 0, self.device, torch.long)
            
        if nonmasked_lengths.sum():
            all_cot_removed_in_batch = False

        batch = {
            "input_ids": input_ids_new,
            "labels": labels_new,
            "position_ids": position_ids_new,
            "full_input_ids": full_input_ids_new,
            "full_labels": full_labels_new,
            "full_position_ids": full_position_ids_new
        }
        
        return batch, all_cot_removed_in_batch
