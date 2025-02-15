import math
import os
import random
import torch
import tqdm

from data import extract_answer
from trainer_base import BaseTrainer
from utils import get_sep_position, batch_ids, save_model


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


class ChunkRemovalTrainer(BaseTrainer):
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args, start_id):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)

        self.remove_by_schedule = args.remove_by_schedule
        self.chunk_removal_schedule = args.removal_schedule
        self.schedule_index = 0

        self.remove_when_flat_loss = args.remove_when_flat_loss
        self.flat_loss_threshold = 0.05

        self.start_id = start_id
        self.chunk_size = args.chunk_size

        self.remove_step_by_step = args.remove_chunks_step_by_step
        self.n_chunks_to_remove = self.chunk_removal_schedule[self.schedule_index][1]

        # USED WITH REMOVE_WHEN_FLAT_LOSS:
        if self.remove_when_flat_loss and args.n_chunks_to_remove_from_start > 0:
            print (f'the number of removed CoT chunks starts from {args.n_chunks_to_remove_from_start}')
            self.scheduled_to_remove = args.n_chunks_to_remove_from_start
        
        self.val_truncation_kwargs = {
            "mask_new_tokens_in_labels": True
        }
        self.val_generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "stop_on_two_eos": True,
            "use_new_tokens": True,
            "new_tokens_start_id": self.start_id
        }

        self.setup_metrics(additional_metrics=["n_chunks_to_remove"])
    
    def _train_process(self):
        step = 0
        if self.writer: self.writer.set_step(step, mode="train")
        loss_log = []

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            if self.writer:
                self.writer.set_step(step, mode="train")
                self.writer.add_scalar("epoch", epoch)
            self.model.train()

            if self.remove_by_schedule and self.schedule_index + 1 < len(self.chunk_removal_schedule):
                if self.chunk_removal_schedule[self.schedule_index + 1][0] == epoch:
                    self.schedule_index += 1
                    self.n_chunks_to_remove = self.chunk_removal_schedule[self.schedule_index][1]
                    if self.args.reset_optimizer and (not all_cot_removed_in_batch):
                        self._reset_optimizer()

            if self.remove_when_flat_loss and len(loss_log) >= 2:
                if torch.abs(loss_log[-1] - loss_log[-2]) < self.flat_loss_threshold:
                    self.n_chunks_to_remove += 1
                    if self.args.reset_optimizer and (not all_cot_removed_in_batch):
                        self._reset_optimizer()

            loss_log.append([])
            print(f"Epoch {epoch}; Step {step}; Scheduled to remove: {self.n_chunks_to_remove} chunks")

            for batch in tqdm.tqdm(self.train_dataloader):
                input_ids, labels, position_ids, all_cot_removed_in_batch = self.process_input_truncation(
                    batch,
                    mask_new_tokens_in_labels=False
                )

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
                    self.metrics_tracker.update("n_chunks_to_remove", self.n_chunks_to_remove)
                
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
            # if self.writer: self.writer.set_step(step, mode="val")
            # accuracy, token_accuracy, ppl = self.evaluate(self.val_dataloader, "val", self.val_truncation_kwargs, self.val_generation_kwargs)

            # if accuracy > best_val_accuracy:
            #     print ('***best so far or removed more CoT tokens***')
            #     best_val_accuracy = accuracy
            #     if self.args.test_path:
            #         if self.writer: self.writer.set_step(step, mode="test")
            #         self.evaluate(self.test_dataloader, "test", self.val_truncation_kwargs, self.val_generation_kwargs)
            
            if epoch % 5 == 0 or epoch == self.args.epochs - 1:
                self.model.save_pretrained(os.path.join(self.args.save_model, f'checkpoint_{epoch}'))


    def process_input_truncation(self, batch, mask_new_tokens_in_labels=False):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        chunk_input_ids = batch['chunk_input_ids'].to(self.device)
        chunk_positions = batch['chunk_positions']

        if self.n_chunks_to_remove == 0:
            eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)
            if (eos_positions != input_ids.shape[1]).any():
                input_ids_new = [
                    input_ids[batch_idx, :eos_positions[batch_idx] + 1]
                    for batch_idx in range(input_ids.shape[0])
                ]
                labels_new = [
                    labels[batch_idx, :eos_positions[batch_idx] + 1]
                    for batch_idx in range(input_ids.shape[0])
                ]
                input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
                labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
                return input_ids_new, labels_new, None, False

            return input_ids, labels, None, False # all_cot_removed_in_batch

        if self.remove_step_by_step:
            return self._step_by_step_chunks_truncation(
                input_ids, labels, chunk_input_ids, chunk_positions,
                self.n_chunks_to_remove, mask_new_tokens_in_labels
            )
        
        return self._random_chunks_truncation(
            input_ids, labels, chunk_input_ids, chunk_positions,
            self.n_chunks_to_remove, mask_new_tokens_in_labels
        )
    
    def _get_sep_positions(self, input_ids):
        first_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)
        return first_sep_positions, second_sep_positions, eos_positions

    @staticmethod
    def same_elements(x):
        return (x == x[0]).all()

    @staticmethod
    def _get_chunk_borders(first_sep_positions, second_sep_positions, chunk_positions, n_chunks_to_remove, batch_idx):
        right_border = second_sep_positions[batch_idx]
        start = first_sep_positions[batch_idx]
        remove_all_chunks = n_chunks_to_remove == len(chunk_positions[batch_idx])
        end = chunk_positions[batch_idx][n_chunks_to_remove] if not remove_all_chunks else right_border
        return start, end

    def _step_by_step_substitution(
            self,
            input_ids,
            labels,
            chunk_input_ids,
            n_chunks_to_remove,
            position_ids,
            start,
            end,
            eos_position,
            mask_new_tokens_in_labels
    ):         
        batch_size = input_ids.shape[0]
        mask_input_ids = []
        mask_labels = []
        
        if n_chunks_to_remove == -1:
            n_chunks_to_remove = chunk_input_ids.shape[-1]

        for i in range(n_chunks_to_remove):
            mask_input_ids.append(torch.full((batch_size, 1), self.start_id).to(self.device))
            mask_input_ids.append(chunk_input_ids[:, [i]])
            mask_labels.append(torch.full((batch_size, 1), -100).to(self.device))
            if mask_new_tokens_in_labels:
                mask_labels.append(torch.full((batch_size, 1), -100).to(self.device))
            else:
                mask_labels.append(chunk_input_ids[:, [i]])
        mask_input_ids = torch.cat(mask_input_ids, dim=-1)
        mask_labels = torch.cat(mask_labels, dim=-1)

        input_ids = torch.cat([
            input_ids[:, :start],
            mask_input_ids,
            input_ids[:, end:eos_position + 1]
        ], dim=-1)
        labels = torch.cat([
            labels[:, :start],
            mask_labels,
            labels[:, end:eos_position + 1]
        ], dim=-1)
        
        if self.args.keep_position:
            position_ids[:, start + 2:] += end - start - 2
        
        return input_ids, labels, position_ids

    def _step_by_step_chunks_truncation(
            self,
            input_ids,
            labels,
            chunk_input_ids,
            chunk_positions,
            n_chunks_to_remove,
            mask_new_tokens_in_labels
    ):
        batch_size = input_ids.shape[0]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)
        nonmasked_lengths = second_sep_positions - first_sep_positions - 1
        
        all_cot_removed_in_batch = True
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1) if self.args.keep_position else None

        if self.same_elements(nonmasked_lengths) and self.same_elements(first_sep_positions):
            # ALL COT HAVE THE SAME SIZE AND POSITION
            start, end = self._get_chunk_borders(
                first_sep_positions, second_sep_positions, chunk_positions,
                n_chunks_to_remove, batch_idx=0
            )
            cot_end = eos_positions[0]
            input_ids, labels, position_ids = self._step_by_step_substitution(
                input_ids, labels, chunk_input_ids, n_chunks_to_remove, position_ids,
                start, end, cot_end, mask_new_tokens_in_labels
            )
            nonmasked_lengths = nonmasked_lengths - (end - start)
        else:
            for batch_idx in range(batch_size):
                start, end = self._get_chunk_borders(
                    first_sep_positions, second_sep_positions, chunk_positions,
                    n_chunks_to_remove, batch_idx=batch_idx
                )
                cot_end = eos_positions[batch_idx]
                input_ids_new, labels_new, position_ids_new = self._step_by_step_substitution(
                    input_ids[[batch_idx]], labels[[batch_idx]], chunk_input_ids[[batch_idx]],
                    n_chunks_to_remove, position_ids[[batch_idx]] if self.args.keep_position else None,
                    start, end, cot_end, mask_new_tokens_in_labels
                )
                input_ids[batch_idx] = input_ids_new.squeeze(0)
                labels[batch_idx] = labels_new.squeeze(0)
                if self.args.keep_position:
                    position_ids[batch_idx] = position_ids_new.squeeze(0)
                nonmasked_lengths[batch_idx] = nonmasked_lengths[batch_idx] - (end - start)
        
        if nonmasked_lengths.sum():
            all_cot_removed_in_batch = False
        
        return input_ids, labels, position_ids, all_cot_removed_in_batch
    
    def _random_chunks_truncation(
            self,
            input_ids,
            labels,
            chunk_input_ids,
            chunk_positions,
            n_chunks_to_remove,
            mask_new_tokens_in_labels
    ):
        batch_size = input_ids.shape[0]
        first_sep_positions, second_sep_positions, eos_positions = self._get_sep_positions(input_ids)
        nonmasked_lengths = second_sep_positions - first_sep_positions - 1
        
        all_cot_removed_in_batch = True
        input_ids_new = []
        labels_new = []
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1) if self.args.keep_position else None

        for batch_idx in range(batch_size):
            right_border = second_sep_positions[batch_idx].item()

            # chunk_positions = list(enumerate(range(left_border, right_border + chunk_size - 1, chunk_size)))
            chunks_enumerated = list(enumerate(chunk_positions[batch_idx]))
            n = min(n_chunks_to_remove, len(chunks_enumerated))
            if n_chunks_to_remove == -1:
                n = len(chunks_enumerated)
            masked_chunks = random.sample(chunks_enumerated, n)

            input_ids_cur = input_ids[batch_idx].detach().clone()
            labels_cur = labels[batch_idx].detach().clone()

            shifts = torch.zeros_like(input_ids_cur)

            for chunk_idx, start in masked_chunks:
                end = min(start + self.chunk_size, right_border)
                shift = shifts[start].item()
                shifts[start:] += end - start - 2
                start, end = start - shift, end - shift

                chunk_id = chunk_input_ids[batch_idx, chunk_idx]
                input_ids_cur = torch.cat((
                    input_ids_cur[:start],
                    torch.tensor(self.start_id).unsqueeze(0).to(self.device),
                    torch.tensor(chunk_id).unsqueeze(0).to(self.device),   # <- we need to place "new-token-start" token first? 
                    input_ids_cur[end:eos_positions[batch_idx] + 1]             # so that model has zero loss on predicting new-token-start,
                ), dim=-1)                                                      # but has loss on predicting the specific new token?

                if mask_new_tokens_in_labels:
                    chunk_id = -100
                
                labels_cur = torch.cat((
                    labels_cur[:start],
                    torch.tensor(-100).unsqueeze(0).to(self.device),
                    torch.tensor(chunk_id).unsqueeze(0).to(self.device),   # <- we need to replace start - 1 with -100
                    labels_cur[end:eos_positions[batch_idx] + 1]            # and start with new-token-id ?
                ), dim=-1)

                if self.args.keep_position:
                    position_ids[batch_idx, start + 2:] += end - start - 2

            input_ids_new.append(input_ids_cur)
            labels_new.append(labels_cur)
            nonmasked_lengths[batch_idx] = nonmasked_lengths[batch_idx] - (end - start)
                
        input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
        
        if nonmasked_lengths.sum():
            all_cot_removed_in_batch = False
            
        return input_ids_new, labels_new, position_ids, all_cot_removed_in_batch
