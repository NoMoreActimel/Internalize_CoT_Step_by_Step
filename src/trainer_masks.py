import math
import os
import random
import torch
import tqdm

from trainer_base import BaseTrainer
from utils import get_sep_position, batch_ids


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


class AuxiliarMasksRemovalTrainer(BaseTrainer):
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)

        # train to produce all masking rates of COTs at once (uniformly sample p per each sample) 
        self.joint_masked_distrubution = args.joint_masked_distribution

        # schedule specific p per each epoch, p = rate of tokens to mask in COT
        self.remove_by_schedule = args.remove_by_schedule

        # list of (from_epoch, p)
        self.masks_removal_schedule = args.removal_schedule
        self.schedule_index = 0

        # p% of tokens are masked from left to right
        self.left_to_right_removal = args.left_to_right_removal
        self.removal_p = self.masks_removal_schedule[self.schedule_index][1]

        self.val_truncation_kwargs = {}
        self.val_generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "stop_on_two_eos": True
        }

        self.setup_metrics(additional_metrics=["removal_p"])
    
    def _train_process(self):
        step = 0
        if self.writer: self.writer.set_step(step, mode="train")
        # best_val_accuracy = float('-inf')
        loss_log = []

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            if self.writer:
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
            if not self.joint_masked_distrubution:
                print_line += f"; Scheduled to remove: {self.removal_p}% of COT"
            print(print_line)

            for batch in tqdm.tqdm(self.train_dataloader):
                input_ids, labels, position_ids, all_cot_removed_in_batch = self.process_input_truncation(batch)

                # if not all_cot_removed_in_batch:
                #     best_val_accuracy = float('-inf')
                if self.args.max_len_train > 0 and input_ids.shape[-1] > self.args.max_len_train:
                    print ('skipped')
                    continue
            
                with self.ctx:
                    if self.args.keep_position:
                        position_ids = position_ids[:, :input_ids.shape[-1]]
                    outputs = self.model.compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)

                loss = outputs.loss
                loss_log[-1].append(loss.item())
                loss.div(self.args.accumulate).backward()
                grad_norm = self.get_grad_norm()

                if step % self.args.accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(self.get_trainable_params(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.metrics_tracker is not None:
                    self.metrics_tracker.update("loss", loss.item())
                    self.metrics_tracker.update("perplexity", loss.exp().item())
                    self.metrics_tracker.update("token_accuracy", outputs.token_accuracy.item())
                    self.metrics_tracker.update("grad_norm", grad_norm)
                    self.metrics_tracker.update("removal_p", self.removal_p)
                
                if step % 100 == 0:
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
            if self.writer: self.writer.set_step(step, mode="val")
            accuracy, token_accuracy, ppl = self.evaluate(self.val_dataloader, "val", self.val_truncation_kwargs, self.val_generation_kwargs)

            # if accuracy > best_val_accuracy:
            #     print ('***best so far or removed more CoT tokens***')
            #     best_val_accuracy = accuracy
            #     if self.args.test_path:
            #         if self.writer: self.writer.set_step(step, mode="test")
            #         self.evaluate(self.test_dataloader, "test", self.val_truncation_kwargs, self.val_generation_kwargs)
            
            if epoch % 5 == 0 or epoch == self.args.epochs - 1:
                self.model.save_pretrained(os.path.join(self.args.save_model, f'checkpoint_{epoch}'))


    def process_input_truncation(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        if self.removal_p == 0.0 and not self.joint_masked_distrubution:
            eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)
            if (eos_positions != input_ids.shape[1]).any():
                input_ids_new = [
                    input_ids[batch_idx, :eos_positions[batch_idx]]
                    for batch_idx in range(input_ids.shape[0])
                ]
                labels_new = [
                    labels[batch_idx, :eos_positions[batch_idx]]
                    for batch_idx in range(input_ids.shape[0])
                ]
                input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
                labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
                return input_ids_new, labels_new, None, False

            return input_ids, labels, None, False # all_cot_removed_in_batch

        if self.left_to_right_removal:
            return self._step_by_step_truncation(input_ids, labels, self.removal_p, self.joint_masked_distrubution)
        
        return self._random_truncation(input_ids, labels, self.removal_p, self.joint_masked_distrubution)
    
    def _get_sep_positions(self, input_ids):
        first_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)
        return first_sep_positions, second_sep_positions, eos_positions

    @staticmethod
    def same_elements(x):
        return (x == x[0]).all()

    def _get_prefix_stepbystep(self, n_tokens_to_remove):
        prefix = self.tokenizer(
            f" ## masked {n_tokens_to_remove} ## ",
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].to(self.device).squeeze(0)

        eos = torch.full((1,), self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
        prefix = torch.cat([prefix, eos])

        ignored_prefix_labels = torch.full_like(prefix, -100)
        return prefix, ignored_prefix_labels

    def _step_by_step_truncation(
            self,
            input_ids,
            labels,
            removal_p,
            joint_masked_distrubution
    ):
        batch_size = input_ids.shape[0]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)
        nonmasked_lengths = second_sep_positions - first_sep_positions - 1
        
        all_cot_removed_in_batch = True
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1) if self.args.keep_position else None

        if self.same_elements(nonmasked_lengths) and self.same_elements(first_sep_positions) and not joint_masked_distrubution:
            # ALL COT HAVE THE SAME SIZE AND POSITION
            n_tokens_to_remove = int(removal_p * nonmasked_lengths[0])

            start = first_sep_positions + 1
            end = start + n_tokens_to_remove
            prefix, ignored_prefix_labels = self._get_prefix_stepbystep(n_tokens_to_remove)
            prefix, ignored_prefix_labels = prefix.unsqueeze(0), ignored_prefix_labels.unsquueze(0)

            input_ids_new = torch.cat([
                input_ids[:, :start - 1],  # move EOS_TOKEN_ID in prefix
                prefix,
                input_ids[:, end:eos_positions + 1]
            ], dim=0)
            labels_new = torch.cat([
                labels[:, :start - 1],
                ignored_prefix_labels,
                labels[:, end:eos_positions + 1]
            ], dim=0)

            position_ids[:, start + len(prefix)] += end - start - 2
            nonmasked_lengths = nonmasked_lengths - (end - start)
        else:
            input_ids_new = []
            labels_new = []
            for batch_idx in range(batch_size):
                if joint_masked_distrubution:
                    removal_p = random.uniform(0, 1)
                n_tokens_to_remove = int(removal_p * nonmasked_lengths[batch_idx])

                start = first_sep_positions[batch_idx] + 1
                end = start + n_tokens_to_remove
                eos = eos_positions[batch_idx]
                prefix, ignored_prefix_labels = self._get_prefix_stepbystep(n_tokens_to_remove)

                input_ids_new.append(torch.cat([
                    input_ids[batch_idx, :start - 1],   # move EOS_TOKEN_ID in prefix
                    prefix,
                    input_ids[batch_idx, end:eos + 1]
                ], dim=0))
                labels_new.append(torch.cat([
                    labels[batch_idx, :start],
                    ignored_prefix_labels,
                    labels[batch_idx, end:eos + 1]
                ], dim=0))
                
                if self.args.keep_position:
                    position_ids[batch_idx, start + len(prefix):] += end - start - 2
                nonmasked_lengths[batch_idx] = nonmasked_lengths[batch_idx] - (end - start)
            
            input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
            labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
            
        if nonmasked_lengths.sum():
            all_cot_removed_in_batch = False
        
        return input_ids_new, labels_new, position_ids, all_cot_removed_in_batch

    @staticmethod
    def _get_random_cot_mask(cot_start, cot_end, n_tokens_to_remove):
        random_indices = torch.randint(cot_start, cot_end, size=n_tokens_to_remove)
        all_indices = torch.arange(cot_start, cot_end)
        mask = ~torch.isin(all_indices, random_indices)
        remaining_indices = all_indices[mask]
        return remaining_indices, random_indices

    def _get_prefix_random_masking(self, cot_start, removed_indices):
        prefix = self.tokenizer(
            f" ## masked tokens {(removed_indices - cot_start).tolist()} ## ",
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].to(self.device).squeeze(0)

        eos = torch.full((1,), self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
        prefix = torch.cat([prefix, eos])

        ignored_prefix_labels = torch.full_like(prefix, -100)
        return prefix, ignored_prefix_labels

    def _random_truncation(
            self,
            input_ids,
            labels,
            removal_p,
            joint_masked_distrubution
    ):
        batch_size = input_ids.shape[0]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)
        nonmasked_lengths = second_sep_positions - first_sep_positions - 1

        all_cot_removed_in_batch = True
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1) if self.args.keep_position else None

        input_ids_new = []
        labels_new = []

        for batch_idx in range(batch_size):
            if joint_masked_distrubution:
                removal_p = random.uniform(0, 1)
            n_tokens_to_remove = int(removal_p * nonmasked_lengths[batch_idx])

            cot_start = first_sep_positions[batch_idx] + 1
            cot_end = second_sep_positions[batch_idx]
            eos = eos_positions[batch_idx]

            remaining_indices, removed_indices = self._get_random_cot_mask(cot_start, cot_end, n_tokens_to_remove)
            prefix, ignored_prefix_labels = self._get_prefix_random_masking(cot_start, removed_indices)

            input_ids_cot_masked = input_ids[batch_idx, remaining_indices]

            input_ids_new.append(torch.cat([
                input_ids[batch_idx, :cot_start - 1],  # move EOS_TOKEN_ID in prefix
                prefix,
                input_ids_cot_masked,
                input_ids[batch_idx, cot_end:eos + 1]
            ]))
            labels_new.append(torch.cat([
                labels[batch_idx, :cot_start - 1],  # move EOS_TOKEN_ID in prefix
                ignored_prefix_labels,
                input_ids_cot_masked,
                labels[batch_idx, cot_end:eos + 1]
            ]))

            if self.args.keep_position:
                mask = torch.zeros_like(input_ids[batch_idx], dtype=torch.long)
                mask[removed_indices] = 1
                mask_cumsum = torch.cumsum(mask)
                position_ids[batch_idx] +=  mask_cumsum

            nonmasked_lengths[batch_idx] = nonmasked_lengths[batch_idx] - n_tokens_to_remove

        input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
            
        if nonmasked_lengths.sum():
            all_cot_removed_in_batch = False
        
        return input_ids_new, labels_new, position_ids, all_cot_removed_in_batch
