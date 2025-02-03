import math
import os
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
            "stop_on_two_eos": True,
        }

    def _train_process(self):
        step = 0
        if self.writer: self.writer.set_step(step, mode="train")

        best_val_accuracy = float('-inf')
        all_cot_removed_in_batch = False

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            if self.writer:
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

                input_ids, labels, position_ids, all_cot_removed_in_batch = self.process_input_truncation(
                    batch, epoch, disable_random_removal_offset=False
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
            
            if self.writer: self.writer.set_step(step, mode="val")
            accuracy, token_accuracy, ppl = self.evaluate(self.val_dataloader, "val", self.val_truncation_kwargs, self.val_generation_kwargs)

            if accuracy > best_val_accuracy:
                print ('***best so far or removed more CoT tokens***')
                best_val_accuracy = accuracy
                if self.args.test_path:
                    if self.writer: self.writer.set_step(step, mode="test")
                    self.evaluate(self.test_dataloader, "test", self.val_truncation_kwargs, self.val_generation_kwargs)
                
            self.model.save_pretrained(os.path.join(self.args.save_model, f'checkpoint_{epoch}'))
    

    def process_input_truncation(self, batch, epoch, disable_random_removal_offset=False):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        if not (self.scheduled_to_remove > 0 or self.args.removal_smoothing_lambda != float('inf')):
            return input_ids, labels, None, False # all_cot_removed_in_batch
        
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

            input_ids_tmp.append(torch.cat((
                input_ids[batch_id, :removal_from_position],
                input_ids[batch_id, removal_to_position:eos_position+1]
            ), dim=-1))
            labels_tmp.append(torch.cat((
                labels[batch_id, :removal_from_position],
                labels[batch_id, removal_to_position:eos_position+1]
            ), dim=-1))
        
        input_ids = batch_ids(input_ids_tmp, self.tokenizer.eos_token_id, self.device, input_ids.dtype)
        labels = batch_ids(labels_tmp, -100, self.device, input_ids.dtype)

        return input_ids, labels, position_ids, all_cot_removed_in_batch
