import math
import os
import torch
import tqdm
import re

from src.trainer.trainer_masks import get_mask_id
from src.trainer.trainer_base import BaseTrainer
from src.utils import get_sep_position, batch_ids


def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    if removal_smoothing_lambda == float('inf'):
        lambda_distribution = torch.zeros(truncate_length)
        lambda_distribution[0] = 1
    else:
        positions = torch.arange(truncate_length)
        lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = lambda_distribution.sum()
        assert cum_prob <= 1
        lambda_distribution[-1] = lambda_distribution[-1] + (1-cum_prob)
    return lambda_distribution


class StepByStepTrainer(BaseTrainer):
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
        self.scheduled_to_remove = 0
        if args.remove_start_from > 0:
            print (f'the number of removed CoT tokens starts from {args.remove_start_from}')
            self.scheduled_to_remove = args.remove_start_from
        
        self.steps_per_epoch = len(self.train_dataloader)
        self.steps_per_removed_token = int(round(self.steps_per_epoch / args.remove_per_epoch))
        self.remove_step_counter = 0

        self.lambda_distribution = compute_lambda_distribution(args.removal_smoothing_lambda)
        # print(self.lambda_distribution.tolist()[:10])

        self.val_truncation_kwargs = {
            "epoch": self.args.pretrain_epochs,
            "disable_random_removal_offset": True
        }
        self.val_generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "insert_const_ids_in_cot": args.replace_mask,
            "stop_on_two_eos": True,
        }

        self.use_prefix = args.use_prefix_stepbystep
        self.prompt_in_percentage = args.prompt_in_percentage
        self.replace_mask = args.replace_mask
        self.replace_mask_in_labels = args.replace_mask_in_labels
        self.mask_id = get_mask_id(self.tokenizer, self.device)

        self.setup_metrics(additional_metrics=["scheduled_to_remove"])

    def _train_process(self):
        if self.args.from_pretrained_checkpoint:
            self._resume_checkpoint(self.args.from_pretrained_checkpoint)
            self.scheduled_to_remove = self.args.remove_per_epoch * self.start_epoch

        batch_size = self.args.batch_size
        step = self.start_epoch * ((len(self.train_dataloader) + batch_size - 1) // batch_size)
        if self.accelerator.is_main_process and self.writer:
            self.writer.set_step(step, mode="train")


        # best_val_accuracy = float('-inf')
        all_cot_removed_in_batch = False

        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="train")
                self.writer.add_scalar("epoch", epoch)
                
            self.model.train()

            if self.scheduled_to_remove < float('inf'):
                self.scheduled_to_remove = int(round(self.scheduled_to_remove))
            if self.scheduled_to_remove >= self.args.remove_all_when_remove_beyond:
                self.scheduled_to_remove = float('inf') # remove all

            print(f"Epoch {epoch}. Scheduled to remove: {self.scheduled_to_remove}")

            for batch in tqdm.tqdm(self.train_dataloader):
                prev_scheduled_to_remove = self.scheduled_to_remove

                if self.remove_step_counter == self.steps_per_removed_token or self.steps_per_removed_token == 0:
                    self.scheduled_to_remove += 1
                    self.remove_step_counter = 0
                if epoch >= self.args.pretrain_epochs:
                    self.remove_step_counter += 1
                
                if self.scheduled_to_remove > prev_scheduled_to_remove:
                    print(f" -epoch {epoch}. step {step}. removing: {self.scheduled_to_remove}")
                    if self.args.reset_optimizer and (not all_cot_removed_in_batch):
                        self._reset_optimizer()
                    
                if self.scheduled_to_remove >= self.args.remove_all_when_remove_beyond:
                    self.scheduled_to_remove = float('inf') # remove all

                batch, all_cot_removed_in_batch = self.process_input_truncation(
                    batch, epoch, disable_random_removal_offset=False
                )
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                position_ids = batch["position_ids"]

                # if not all_cot_removed_in_batch:
                #     best_val_accuracy = float('-inf')

                if self.args.max_len_train > 0 and input_ids.shape[-1] > self.args.max_len_train:
                    print ('skipped')
                    continue
            
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        if self.args.keep_position:
                            position_ids = position_ids[:, :input_ids.shape[-1]]
                        outputs = self.model.compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)

                    loss = outputs.loss
                    self.accelerator.backward(loss.div(self.args.accumulate))
                    grad_norm = self.get_grad_norm()

                    torch.nn.utils.clip_grad_norm_(self.get_trainable_params(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                
                if self.metrics_tracker is not None:
                    self.metrics_tracker.update("loss", loss.item())
                    self.metrics_tracker.update("perplexity", loss.exp().item())
                    self.metrics_tracker.update("token_accuracy", outputs.token_accuracy.item())
                    self.metrics_tracker.update("grad_norm", grad_norm)
                    self.metrics_tracker.update("scheduled_to_remove", self.scheduled_to_remove)

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
            
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="val")
            
            accuracy, token_accuracy, ppl = self.evaluate(
                self.val_dataloader,
                "val",
                self.val_truncation_kwargs,
                self.val_generation_kwargs,
                perform_generative_eval=True,
                generative_eval_hooks=[self.generation_eval_insert_ids_hook]
            )
            save_best = self.check_best(accuracy, token_accuracy, ppl)
            
            if self.args.test_path or (self.args.test_split and self.args.test_split != self.args.val_split):
                if self.accelerator.is_main_process and self.writer:
                    self.writer.set_step(step, mode="test")
                accuracy, token_accuracy, ppl = self.evaluate(
                    self.test_dataloader,
                    "test",
                    self.val_truncation_kwargs,
                    self.val_generation_kwargs,
                    perform_generative_eval=True,
                    generative_eval_hooks=[self.generation_eval_insert_ids_hook]
                )

            # if accuracy > best_val_accuracy:
            #     print ('***best so far or removed more CoT tokens***')
            #     best_val_accuracy = accuracy
            #     if self.args.test_path:
            #         if self.writer: self.writer.set_step(step, mode="test")
            #         self.evaluate(self.test_dataloader, "test", self.val_truncation_kwargs, self.val_generation_kwargs)
            #     self._save_checkpoint(epoch=self.epoch, save_best=True, only_best=True)
            self.save_epoch(epoch, save_best=save_best)


    def generation_eval_insert_ids_hook(self, batch, truncation_kwargs, generation_kwargs):
        if generation_kwargs.get("insert_const_ids_in_cot", False):
            ids_to_insert, insert_position = self.get_stepbystep_mask_for_gen_eval(batch)
            generation_kwargs["ids_to_insert"] = ids_to_insert
            generation_kwargs["insert_position"] = insert_position
        return generation_kwargs

    def _extract_mask_positions_stepbystep(self, input_ids):
        sample = self.tokenizer.batch_decode(input_ids)[0]
        m = re.search(r'##\s*(.*?)\s*##', sample)
        removed_part = m.group(1).strip() if m else None
        removed_part_split = removed_part.split(' ')
        n_tokens_to_remove = int(removed_part_split[1])
        return n_tokens_to_remove
    
    # instead of sampling mask, extract it from input_ids to match the added ## removed ## prefix
    def get_stepbystep_mask_for_gen_eval(self, batch):
        input_ids = batch["input_ids"]
        if input_ids.shape[0] != 1:
            print("\nUsing get_contiguous_mask_for_gen_eval with input_ids.shape[0] > 1,")
            print("ensure the same mask is applied in process_input_truncation for all samples!\n")
        
        insert_position = 0
        n_tokens_to_remove = self._extract_mask_positions_stepbystep(input_ids)

        mask_shape = (input_ids.shape[0], n_tokens_to_remove)
        ids_to_insert = torch.full(mask_shape, self.mask_id.item(), dtype=torch.long, device=input_ids.device)
        return ids_to_insert, insert_position

    def _get_prefix_stepbystep(self, to_remove, cot_length):
        # Ensure Python scalars for formatting/division
        if torch.is_tensor(to_remove):
            to_remove = to_remove.item()
        if torch.is_tensor(cot_length):
            cot_length = cot_length.item()
        to_remove = int(to_remove)
        cot_length = int(cot_length)
        prefix = ""

        if self.prompt_in_percentage:
            pct = (round(to_remove / cot_length) * 100) if cot_length > 0 else 0
            prefix_str = f" ## masked {pct} % of CoT ## "
        else:
            prefix_str = f" ## masked {to_remove} ## "

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
        if self.use_prefix:
            ids_to_cat = [ids[..., :cot_start - 1], prefix_ids] # move EOS_TOKEN_ID in prefix
        else:
            ids_to_cat = [ids[..., :cot_start]]
        if cot_start < start:
            ids_to_cat.append(ids[..., cot_start:start])

        if self.args.replace_mask:
            if labels_flag and (self.replace_mask_in_labels is False):
                # Labels: recover all COT tokens in labels in case of mask replacement
                ids_to_cat.append(ids[..., start:end])
            else:
                mask_tensor = self._fill_tensor_like_with_last_dim(ids, self.mask_id.item(), (end - start).item())
                ids_to_cat.append(mask_tensor)
        
        ids_to_cat.append(ids[..., end:eos + 1])

        ids_new = torch.cat(ids_to_cat, dim=dim)
        return ids_new

    def process_input_truncation(self, batch, epoch, disable_random_removal_offset=False):
        input_ids = batch['input_ids']
        labels = batch['labels']

        if not (self.scheduled_to_remove > 0 or self.args.removal_smoothing_lambda != float('inf')):
            position_ids = None
            if self.args.keep_position:
                batch_size = input_ids.shape[0]
                position_ids = torch.arange(
                    0, input_ids.shape[-1], dtype=torch.long, device=self.device
                ).unsqueeze(0).repeat(batch_size, 1)
            return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}, False
        
        batch_size = input_ids.shape[0]

        first_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)

        all_cot_removed_in_batch = False
        input_ids_tmp = []
        labels_tmp = []

        # compute number of tokens to remove
        random_removal_offset = torch.multinomial(self.lambda_distribution, batch_size, replacement=True).to(self.device)
        if disable_random_removal_offset:
            random_removal_offset.fill_(0)
        
        to_remove = self.scheduled_to_remove + random_removal_offset
        if epoch < self.args.pretrain_epochs:
            to_remove.fill_(self.args.remove_start_from)
        
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1) if self.args.keep_position else None
        
        if self.args.removal_side == 'left':
            removal_from_positions = first_sep_positions + 1 # remove from, including
            removal_to_positions = first_sep_positions + 1 + to_remove # remove to, not including
        else: # removal_side == 'right'
            removal_to_positions = second_sep_positions
            removal_from_positions = second_sep_positions - to_remove

        all_cot_removed_in_batch = True
        for batch_id in range(input_ids.shape[0]):
            eos_position = eos_positions[batch_id]
            removal_from_position = removal_from_positions[batch_id]
            removal_to_position = removal_to_positions[batch_id]

            removal_from_position = max(removal_from_position, first_sep_positions[batch_id]+1)
            if removal_to_position < second_sep_positions[batch_id]:
                all_cot_removed_in_batch = False
            removal_to_position = min(removal_to_position, second_sep_positions[batch_id])

            if self.args.keep_position:
                position_ids[batch_id, removal_from_position-1:] += removal_to_position - removal_from_position

            cot_start = first_sep_positions[batch_id] + 1
            start = cot_start
            end = start + removal_to_position - removal_from_position
            eos = eos_positions[batch_id]

            prefix, ignored_prefix_labels = None, None
            if self.use_prefix:
                prefix, ignored_prefix_labels = self._get_prefix_stepbystep(
                    to_remove=to_remove[batch_id].item(),
                    cot_length=removal_to_position - removal_from_position
                )

            input_ids_new = self._concat_ids(input_ids[batch_id], prefix, cot_start, start, end, eos, dim=0)
            labels_new = self._concat_ids(labels[batch_id], ignored_prefix_labels, cot_start, start, end, eos, dim=0, labels_flag=True)

            input_ids_new_legacy = torch.cat((
                input_ids[batch_id, :removal_from_position],
                input_ids[batch_id, removal_to_position:eos_position+1]
            ), dim=-1)
            labels_new_legacy = torch.cat((
                labels[batch_id, :removal_from_position],
                labels[batch_id, removal_to_position:eos_position+1]
            ), dim=-1)

            if not self.use_prefix and not self.replace_mask:
                assert torch.all(input_ids_new == input_ids_new_legacy)
            if not self.use_prefix and not self.replace_mask_in_labels:
                assert torch.all(labels_new == labels_new_legacy)

            input_ids_tmp.append(input_ids_new)
            labels_tmp.append(labels_new)
        
        input_ids = batch_ids(input_ids_tmp, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels = batch_ids(labels_tmp, -100, self.device, input_ids.dtype)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids
        }

        return batch, all_cot_removed_in_batch
