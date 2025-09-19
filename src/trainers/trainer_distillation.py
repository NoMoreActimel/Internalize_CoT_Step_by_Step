import torch
import torch.nn.functional as F
import tqdm

from src.trainers.trainer_simple import SimpleTrainer

class DistillationTrainer(SimpleTrainer):
    """
    Trainer for distilling student model from teacher generations and logits.
    
    This trainer inherits from SimpleTrainer and adds knowledge distillation
    capabilities:
    - Trains student on teacher-generated sequences
    - Uses both cross-entropy loss and KL divergence with teacher logits
    - Supports temperature scaling for distillation
    """
    
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, 
                 test_dataloader, use_fused, args):
        super().__init__(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, 
                        test_dataloader, use_fused, args)
        
        # Distillation hyperparameters
        self.temperature = getattr(args, 'distillation_temperature', 4.0)
        self.alpha = getattr(args, 'distillation_alpha', 0.7)  # Weight for KL loss
        self.beta = getattr(args, 'distillation_beta', 0.3)   # Weight for CE loss
        
        print(f"Distillation settings:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Alpha (KL weight): {self.alpha}")
        print(f"  Beta (CE weight): {self.beta}")

        additional_metrics = ["total_loss", "ce_loss", "distillation_loss"]
        if self.jepa_training:
            additional_metrics.append("CE_loss")
            additional_metrics.append("JEPA_loss")
        self.setup_metrics(additional_metrics=additional_metrics)
    
    def _train_process(self):
        if self.args.from_pretrained_checkpoint:
            self._resume_checkpoint(self.args.from_pretrained_checkpoint)

        batch_size = self.args.batch_size
        step = self.start_epoch * ((len(self.train_dataloader) + batch_size - 1) // batch_size)
        if self.accelerator.is_main_process and self.writer:
            self.writer.set_step(step, mode="train")

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
                teacher_logits = batch.get("teacher_logits", None)

                if self.args.max_len_train > 0 and input_ids.shape[-1] > self.args.max_len_train:
                    print('skipped')
                    continue
            
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        if self.args.keep_position:
                            position_ids = position_ids[:, :input_ids.shape[-1]]
                        
                        # Get student model outputs
                        outputs = self.model.compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)
                        
                        # Standard cross-entropy loss
                        ce_loss = outputs.loss
                        
                        # Compute distillation loss if teacher logits are available
                        if teacher_logits is not None:
                            distillation_loss = self.compute_distillation_loss(
                                student_logits=outputs.logits,
                                teacher_logits=teacher_logits,
                                labels=labels
                            )
                            
                            # Combine losses
                            total_loss = self.beta * ce_loss + self.alpha * distillation_loss
                        else:
                            total_loss = ce_loss
                            distillation_loss = torch.tensor(0.0)

                    loss_log[-1].append(total_loss.item())
                    self.accelerator.backward(total_loss.div(self.args.accumulate))
                    grad_norm = self.get_grad_norm()

                    torch.nn.utils.clip_grad_norm_(self.get_trainable_params(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.metrics_tracker is not None:
                    self.metrics_tracker.update("total_loss", total_loss.item())
                    self.metrics_tracker.update("ce_loss", ce_loss.item())
                    self.metrics_tracker.update("distillation_loss", distillation_loss.item())
                    self.metrics_tracker.update("perplexity", ce_loss.exp().item())
                    self.metrics_tracker.update("token_accuracy", outputs.token_accuracy.item())
                    self.metrics_tracker.update("grad_norm", grad_norm)
                
                if self.accelerator.is_main_process and step % 100 == 0:
                    token_accuracy = outputs.token_accuracy.item()
                    ppl = ce_loss.exp().item()
                    print(f"Step: {step}. Total Loss: {total_loss.item():.4f}. "
                          f"CE Loss: {ce_loss.item():.4f}. "
                          f"KL Loss: {distillation_loss.item():.4f}. "
                          f"PPL: {ppl:.4f}. Token Accuracy: {token_accuracy:.4f}")
                    
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

    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute the knowledge distillation loss between student and teacher logits.
        
        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]  
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            KL divergence loss between teacher and student distributions
        """
        # Only compute loss where teacher_logits is not all padding (e.g., -100) along vocab dimension
        mask = (labels != -100) & ~(teacher_logits != -100.0).all(dim=-1)
        
        if not mask.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Apply temperature scaling
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = teacher_logits / self.temperature
        
        # Compute softmax probabilities
        student_probs = F.log_softmax(student_logits_scaled, dim=-1)
        teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
        
        # Compute KL divergence only on non-masked positions
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        kl_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='none'
        ).sum(dim=-1)
        
        # Apply mask and take mean
        kl_loss = kl_loss.view(mask.shape)
        kl_loss = (kl_loss * mask.float()).sum() / mask.float().sum()
        
        # Scale by temperature squared (standard in distillation)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return kl_loss
    
    def process_input_truncation(self, batch):
        """
        Override to handle teacher logits in the batch.
        """
        batch["position_ids"] = None
        if self.args.keep_position:
            position_ids = torch.arange(0, batch["input_ids"].shape[-1], dtype=torch.long, device=self.device)
            if batch['input_ids'].ndim == 2:
                position_ids = position_ids.unsqueeze(0).repeat(batch['input_ids'].shape[0], 1)
            batch["position_ids"] = position_ids
        
        # Move teacher logits to device if present
        if "teacher_logits" in batch:
            batch["teacher_logits"] = batch["teacher_logits"].to(self.device)
        
        return batch, False
