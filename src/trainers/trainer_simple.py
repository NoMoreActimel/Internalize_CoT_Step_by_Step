import torch
import tqdm

from trainers.trainer_base import BaseTrainer


class SimpleTrainer(BaseTrainer):
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)

        self.val_truncation_kwargs = {}
        self.val_generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "stop_on_two_eos": True,
            "position_ids_shift": False
        }

        # Metrics
        self.setup_metrics()
    
    def _train_process(self):
        if self.args.from_pretrained_checkpoint:
            self._resume_checkpoint(self.args.from_pretrained_checkpoint)

        batch_size = self.args.batch_size
        step = self.start_epoch * ((len(self.train_dataloader) + batch_size - 1) // batch_size)
        if self.accelerator.is_main_process and self.writer:
            self.writer.set_step(step, mode="train")

        # best_val_accuracy = float('-inf')
        loss_log = []

        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="train")
                self.writer.add_scalar("epoch", epoch)
            self.model.train()

            loss_log.append([])

            print_line = f"Epoch {epoch}; Step {step}"
            print(print_line)

            for batch in tqdm.tqdm(self.train_dataloader):
                batch, all_cot_removed_in_batch = self.process_input_truncation(batch)
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
                    loss_log[-1].append(loss.item())
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
            
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="val")
            
            acc, tok_acc, ppl = self.evaluate(
                dataloader=self.val_dataloader,
                name="val",
                truncation_kwargs=self.val_truncation_kwargs,
                generation_kwargs=self.val_generation_kwargs,
                perform_generative_eval=True
            )
            save_best = self.check_best(acc, tok_acc, ppl)
            if self.args.test_path or (self.args.test_split and self.args.test_split != self.args.val_split):
                if self.accelerator.is_main_process and self.writer:
                    self.writer.set_step(step, mode="test")
                accuracy, token_accuracy, ppl = self.evaluate(
                    self.test_dataloader,
                    "test",
                    self.val_truncation_kwargs,
                    self.val_generation_kwargs,
                    perform_generative_eval=True
                )

            self.save_epoch(epoch)

    def process_input_truncation(self, batch):
        batch["position_ids"] = None
        if self.args.keep_position:
            position_ids = torch.arange(0, batch["input_ids"].shape[-1], dtype=torch.long, device=self.device)
            if batch['input_ids'].ndim == 2:
                position_ids = position_ids.unsqueeze(0).repeat(batch['input_ids'].shape[0], 1)
            batch["position_ids"] = position_ids
        
        return batch, False
