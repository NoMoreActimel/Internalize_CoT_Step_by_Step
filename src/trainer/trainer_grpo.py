import copy
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

from src.trainer.trainer_masks import AuxiliarMasksRemovalTrainer
from src.utils import (
    get_sep_position, extract_answer, batch_ids,
    DoubleEOSLogitsProcessor, DoubleEOSStoppingCriteria
)


class GRPOMasksTrainer(AuxiliarMasksRemovalTrainer):
    """GRPO trainer that samples completions from the model, scores them,
    and trains using group-relative policy optimization with CoT masking.

    Training loop per step:
      1. Extract prompts (questions) from the original dataset batch
      2. Generate G completions per prompt via temperature sampling
      3. Score completions by answer correctness (reward 0/1)
      4. Compute group-relative advantages
      5. Build SFT-format batches from the completions
      6. Apply process_input_truncation (same masking as SFT training)
      7. Compute GRPO loss (clipped surrogate + KL penalty) and update
    """

    def __init__(self, model, optimizer, tokenizer, device,
                 train_dataloader, val_dataloader, test_dataloader,
                 use_fused, args):
        super().__init__(
            model, optimizer, tokenizer, device,
            train_dataloader, val_dataloader, test_dataloader,
            use_fused, args
        )

        # GRPO hyperparameters (with safe defaults)
        self.grpo_group_size = getattr(args, 'grpo_group_size', 4)
        self.grpo_clip_eps = getattr(args, 'grpo_clip_eps', 0.2)
        self.grpo_kl_coeff = getattr(args, 'grpo_kl_coeff', 0.01)
        self.grpo_temperature = getattr(args, 'grpo_temperature', 0.7)
        self.grpo_max_new_tokens = getattr(args, 'grpo_max_new_tokens', None) or self.args.max_new_tokens

        # Frozen reference model for KL penalty
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ref_model = copy.deepcopy(unwrapped)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model = self.ref_model.to(self.device)

    # ------------------------------------------------------------------
    # Prompt / answer extraction
    # ------------------------------------------------------------------
    def _extract_prompts_and_answers(self, batch):
        """Return (prompts, gt_answers) from a dataset batch.

        Each prompt is input_ids[:first_eos+1].
        Ground-truth answer is extracted with extract_answer on the text
        after the first EOS (same convention as evaluation).
        """
        input_ids = batch['input_ids']
        first_sep = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=0)

        prompts = []
        gt_answers = []
        for i in range(input_ids.shape[0]):
            prompts.append(input_ids[i, :first_sep[i] + 1])

            tgt_text = self.tokenizer.decode(
                input_ids[i, first_sep[i] + 1:], skip_special_tokens=True
            )
            gt_answers.append(extract_answer(tgt_text))

        return prompts, gt_answers

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_completions(self, prompts):
        """Sample G completions per prompt using temperature sampling.

        Returns list[list[Tensor]], shape [n_prompts][G], each tensor is
        the full sequence (prompt + generated tokens).
        """
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        all_completions = []

        for prompt in prompts:
            prompt_batch = prompt.unsqueeze(0).expand(self.grpo_group_size, -1)

            generation_config = GenerationConfig.from_model_config(
                unwrapped.base_model.config
            )
            if hasattr(generation_config, 'pad_token_id'):
                generation_config.pad_token_id = self.tokenizer.eos_token_id
            generation_config.eos_token_id = -1  # rely on stopping criteria

            logits_processor = LogitsProcessorList([
                DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)
            ])
            stopping_criteria = StoppingCriteriaList([
                DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)
            ])

            with self.accelerator.autocast():
                outputs = unwrapped.base_model.generate(
                    input_ids=prompt_batch,
                    generation_config=generation_config,
                    max_new_tokens=self.grpo_max_new_tokens,
                    do_sample=True,
                    temperature=self.grpo_temperature,
                    top_p=0.95,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )

            all_completions.append([outputs[j] for j in range(self.grpo_group_size)])

        return all_completions

    # ------------------------------------------------------------------
    # Scoring & advantages
    # ------------------------------------------------------------------
    def _score_completions(self, completions, gt_answers):
        """Binary reward: 1 if extracted answer matches ground truth, else 0."""
        rewards = []
        for group, gt_ans in zip(completions, gt_answers):
            group_rewards = []
            for completion in group:
                eos_pos = (completion == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    text = self.tokenizer.decode(
                        completion[eos_pos[0] + 1:], skip_special_tokens=True
                    )
                else:
                    text = self.tokenizer.decode(completion, skip_special_tokens=True)

                pred_ans = extract_answer(text)
                group_rewards.append(1.0 if pred_ans == gt_ans else 0.0)
            rewards.append(group_rewards)
        return rewards

    @staticmethod
    def _compute_advantages(rewards):
        """Group-relative advantage normalisation."""
        advantages = []
        for group_rewards in rewards:
            r = torch.tensor(group_rewards, dtype=torch.float32)
            std_r = r.std()
            if std_r < 1e-8:
                advantages.append(torch.zeros_like(r))
            else:
                advantages.append((r - r.mean()) / (std_r + 1e-8))
        return advantages

    # ------------------------------------------------------------------
    # Build SFT batch from generated completions
    # ------------------------------------------------------------------
    def _is_valid_completion(self, completion):
        """A valid completion has >= 3 EOS tokens (question, cot, answer)."""
        return (completion == self.tokenizer.eos_token_id).sum().item() >= 3

    def _build_sft_batch(self, completions):
        """Pack a list of 1-D completion tensors into a padded batch dict
        with labels (question part masked with -100)."""
        input_ids_list = []
        labels_list = []

        for comp in completions:
            ids = comp.clone()
            lbl = comp.clone()

            eos_pos = (ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                lbl[:eos_pos[0] + 1] = -100  # mask question + first EOS

            input_ids_list.append(ids)
            labels_list.append(lbl)

        dtype = completions[0].dtype
        input_ids = batch_ids(input_ids_list, self.tokenizer.eos_token_id, self.device, dtype)
        labels = batch_ids(labels_list, -100, self.device, dtype)
        return {"input_ids": input_ids, "labels": labels}

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------
    def _compute_per_token_log_probs(self, model, batch):
        """Return (token_log_probs, mask) both [B, T-1].

        token_log_probs[b,t] = log π(label[b,t+1] | input[b,:t+1])
        mask[b,t] = 1 iff label[b,t+1] >= 0 (i.e. not masked)
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = model(input_ids=input_ids).logits          # [B, T, V]
        shift_logits = logits[:, :-1, :]                     # [B, T-1, V]
        shift_labels = labels[:, 1:]                         # [B, T-1]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        mask = shift_labels.ge(0).float()
        token_log_probs = token_log_probs * mask
        return token_log_probs, mask

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------
    def _grpo_loss(self, log_probs, old_log_probs, ref_log_probs,
                   mask, advantages):
        """Clipped surrogate objective + KL divergence penalty.

        Returns (loss, mean_kl) scalars.
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio,
                              1 - self.grpo_clip_eps,
                              1 + self.grpo_clip_eps)

        adv = advantages.unsqueeze(-1)                       # [B, 1]
        policy_loss = -torch.min(ratio * adv, clipped * adv)

        # Unbiased KL estimator: KL(π_ref ‖ π_θ)
        log_r = ref_log_probs - log_probs
        kl = torch.exp(log_r) - log_r - 1

        per_token = policy_loss + self.grpo_kl_coeff * kl
        per_sample = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)

        return per_sample.mean(), (kl * mask).sum() / mask.sum().clamp(min=1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def _train_process(self):
        step = 0

        if self.args.from_pretrained_checkpoint:
            step = self._resume_checkpoint(
                self.args.from_pretrained_checkpoint,
                load_optimizer_state=not self.args.resume_without_optimizer
            )
        if self.args.from_pretrained_fullcot_checkpoint:
            step = self._resume_checkpoint(
                self.args.from_pretrained_fullcot_checkpoint,
                load_optimizer_state=False
            )

        if self.accelerator.is_main_process and self.writer:
            self.writer.set_step(step, mode="train")

        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="train")
                self.writer.add_scalar("epoch", epoch)

            # Removal-rate schedule (inherited from masks trainer)
            if self.remove_by_schedule \
                    and self.schedule_index + 1 < len(self.masks_removal_schedule):
                if self.masks_removal_schedule[self.schedule_index + 1][0] == epoch:
                    self.schedule_index += 1
                    self.removal_p = self.masks_removal_schedule[self.schedule_index][1]

            loss_log = []
            print_line = f"Epoch {epoch}; Step {step}"
            if not self.joint_masked_distribution:
                print_line += f"; Scheduled to remove: {self.removal_p * 100}% of COT"
            print(print_line)

            if self.accelerator.is_main_process:
                print(f">>> Using mixed precision: {self.accelerator.state.mixed_precision}")

            for batch_idx, batch in tqdm.tqdm(enumerate(self.train_dataloader)):
                # ---- generation phase (no grad) ----
                self.model.eval()

                prompts, gt_answers = self._extract_prompts_and_answers(batch)
                completions = self._generate_completions(prompts)
                rewards = self._score_completions(completions, gt_answers)
                advantages = self._compute_advantages(rewards)

                all_rewards = [r for grp in rewards for r in grp]
                mean_reward = np.mean(all_rewards) if all_rewards else 0.0

                # flatten & filter
                comp_flat, adv_flat = [], []
                for grp_c, grp_a in zip(completions, advantages):
                    for c, a in zip(grp_c, grp_a):
                        if self._is_valid_completion(c):
                            comp_flat.append(c)
                            adv_flat.append(a)

                if not comp_flat:
                    step += 1
                    continue

                adv_tensor = torch.stack(adv_flat).to(self.device)

                # skip when all advantages are zero (no contrast in any group)
                if adv_tensor.abs().max() < 1e-8:
                    if self.accelerator.is_main_process and step % 100 == 0:
                        print(f"Step {step}: no learning signal "
                              f"(mean_reward={mean_reward:.2f}), skipping")
                    step += 1
                    continue

                # ---- build SFT batch & apply masking ----
                sft_batch = self._build_sft_batch(comp_flat)
                truncated_batch, _ = self.process_input_truncation(sft_batch)

                if self.args.max_len_train > 0 \
                        and truncated_batch["input_ids"].shape[-1] > self.args.max_len_train:
                    print("skipped (too long)")
                    step += 1
                    continue

                # ---- training phase ----
                self.model.train()

                with torch.no_grad():
                    with self.accelerator.autocast():
                        old_lp, mask = self._compute_per_token_log_probs(
                            self.model, truncated_batch)
                        ref_lp, _   = self._compute_per_token_log_probs(
                            self.ref_model, truncated_batch)

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        cur_lp, mask = self._compute_per_token_log_probs(
                            self.model, truncated_batch)
                        loss, mean_kl = self._grpo_loss(
                            cur_lp, old_lp, ref_lp, mask, adv_tensor)

                    loss_log.append(loss.item())
                    self.accelerator.backward(loss.div(self.args.accumulate))
                    grad_norm = self.get_grad_norm()

                    torch.nn.utils.clip_grad_norm_(
                        self.get_trainable_params(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # ---- logging ----
                if self.metrics_tracker is not None:
                    self.metrics_tracker.update("loss", loss.item())
                    self.metrics_tracker.update("grad_norm", grad_norm)
                    self.metrics_tracker.update("removal_p", self.removal_p)

                if self.accelerator.is_main_process and step % 100 == 0:
                    print(f"Step: {step}. Loss: {loss.item():.4f}. "
                          f"Mean reward: {mean_reward:.3f}. "
                          f"KL: {mean_kl.item():.4f}")
                    if self.metrics_tracker is not None:
                        self.writer.set_step(step, mode="train")
                        self.log_scalars(self.metrics_tracker)
                        self.metrics_tracker.reset()
                        self.writer.add_scalar(
                            "learning rate",
                            self.optimizer.param_groups[0]['lr'])
                        self.writer.add_scalar("mean_reward", mean_reward)
                        self.writer.add_scalar("kl_divergence", mean_kl.item())

                step += 1

            if loss_log:
                avg = sum(loss_log) / len(loss_log)
                print(f"Epoch {epoch} avg loss: {avg:.4f}")

            save_best = self.evaluate(step, base_name="val")
            if self.args.test_path or \
                    (self.args.test_split and self.args.test_split != self.args.val_split):
                self.evaluate(step, base_name="test")

            self.save_epoch(epoch, save_best=save_best)
