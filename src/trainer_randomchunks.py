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


class RandomChunksTrainer(BaseTrainer):
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args, start_id):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)

        self.remove_by_schedule = args.remove_by_schedule
        self.chunk_removal_schedule = args.chunk_removal_schedule
        self.schedule_index = 0

        self.remove_when_flat_loss = args.remove_when_flat_loss
        self.flat_loss_threshold = 0.05

        self.start_id = start_id
        self.chunk_size = args.chunk_size
        self.n_chunks_to_remove = self.chunk_removal_schedule[self.schedule_index][1]

        # USED WITH REMOVE_WHEN_FLAT_LOSS:
        if self.remove_when_flat_loss and args.n_chunks_to_remove_from_start > 0:
            print (f'the number of removed CoT chunks starts from {args.n_chunks_to_remove_from_start}')
            self.scheduled_to_remove = args.n_chunks_to_remove_from_start
        
        self.val_truncation_kwargs = {
            "n_chunks_to_remove": self.n_chunks_to_remove,
            "mask_new_tokens_in_labels": True
        }
        self.val_generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "stop_on_two_eos": True,
            "use_new_tokens": True,
            "new_tokens_start_id": self.start_id
        }
    
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
                if loss_log[-1] - loss_log[-2] < self.flat_loss_threshold:
                    self.n_chunks_to_remove += 1
                    if self.args.reset_optimizer and (not all_cot_removed_in_batch):
                        self._reset_optimizer()

            loss_log.append([])
            print(f"Epoch {epoch}; Step {step}; Scheduled to remove: {self.n_chunks_to_remove} chunks")

            for batch in tqdm.tqdm(self.train_dataloader):
                input_ids, labels, position_ids, all_cot_removed_in_batch = self.process_input_truncation(
                    batch,
                    n_chunks_to_remove=self.n_chunks_to_remove,
                    mask_new_tokens_in_labels=False
                )

                if not all_cot_removed_in_batch:
                    best_val_accuracy = float('-inf')

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

                if self.metrics_tracker:
                    self.metrics_tracker.update("loss", loss.item())
                    self.metrics_tracker.update("perplexity", loss.exp().item())
                    self.metrics_tracker.update("token_accuracy", outputs.token_accuracy.item())
                    self.metrics_tracker.update("grad_norm", grad_norm)
                
                if step % 100 == 0:
                    token_accuracy = outputs.token_accuracy.item()
                    ppl = loss.exp().item()
                    print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                    if self.metrics_tracker:
                        self.log_scalars(self.metrics_tracker)
                        self.metrics_tracker.reset()
                        self.writer.add_scalar("learning rate", self.optimizer.param_groups[0]['lr'])

                step += 1

            loss_log[-1] = sum(loss_log[-1]) / len(loss_log[-1])
            if self.writer: self.writer.set_step(step, mode="val")
            accuracy, token_accuracy, ppl = self.evaluate(self.val_dataloader, "val", self.val_truncation_kwargs, self.val_generation_kwargs)

            if accuracy > best_val_accuracy:
                print ('***best so far or removed more CoT tokens***')
                best_val_accuracy = accuracy
                if self.args.test_path:
                    if self.writer: self.writer.set_step(step, mode="test")
                    self.evaluate(self.test_dataloader, "test", self.val_truncation_kwargs, self.val_generation_kwargs)

            self.model.save_pretrained(os.path.join(self.args.save_model, f'checkpoint_{epoch}'))


    def process_input_truncation(self, batch, n_chunks_to_remove, mask_new_tokens_in_labels=False):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        chunk_input_ids = batch['chunk_input_ids'].to(self.device)
        chunk_positions = batch['chunk_positions']

        if n_chunks_to_remove == 0:
            return input_ids, labels, None, False # all_cot_removed_in_batch
        
        batch_size = input_ids.shape[0]

        first_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=2)

        nonmasked_lenghts = second_sep_positions - first_sep_positions - 1
        
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
                    torch.tensor(self.start_id).unsqueeze(0).to(input_ids.device),
                    torch.tensor(chunk_id).unsqueeze(0).to(input_ids.device),   # <- we need to place "new-token-start" token first? 
                    input_ids_cur[end:eos_positions[batch_idx] + 1]             # so that model has zero loss on predicting new-token-start,
                ), dim=-1)                                                      # but has loss on predicting the specific new token?

                if mask_new_tokens_in_labels:
                    chunk_id = -100
                
                labels_cur = torch.cat((
                    labels_cur[:start - 1],
                    torch.tensor(-100).unsqueeze(0).to(input_ids.device),
                    torch.tensor(chunk_id).unsqueeze(0).to(input_ids.device),   # <- we need to replace start - 1 with -100
                    labels_cur[end - 1:eos_positions[batch_idx] + 1]            # and start with new-token-id ?
                ), dim=-1)

                if self.args.keep_position:
                    position_ids[batch_idx, start + 2:] += end - start - 2

            input_ids_new.append(input_ids_cur)
            labels_new.append(labels_cur)
            nonmasked_lenghts[batch_idx] = nonmasked_lenghts[batch_idx] - (end - start)
                
        input_ids_new = batch_ids(input_ids_new, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels_new = batch_ids(labels_new, -100, self.device, input_ids.dtype)
        
        if nonmasked_lenghts.sum():
            all_cot_removed_in_batch = False
            
        return input_ids_new, labels_new, position_ids, all_cot_removed_in_batch
