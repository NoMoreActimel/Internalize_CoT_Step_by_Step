import copy
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from src.trainer.trainer_masks import AuxiliarMasksRemovalTrainer
from src.utils import get_sep_position, extract_answer, batch_ids


class GRPOMasksTrainer(AuxiliarMasksRemovalTrainer):
    """GRPO trainer that samples completions from the model, scores them,
    and trains using group-relative policy optimization with CoT masking.

    Per outer iteration (one dataloader batch):
      1. Build rollout prompts at R=0:  Q + " ## masked 0 % of CoT ## " + EOS
         (in-distribution for the SFT'd auxiliary-masks model — identical
          prefix string to what random_masking produces at R=0).
      2. Sample G completions per prompt via temperature sampling through
         ImplicitModel.sample_completions (same entry point as SFT eval).
      3. Score by answer correctness; compute group-relative advantages.
      4. Sync old_model ← current policy (frozen π_θ_old snapshot).
      5. For k = 1..K (= --grpo_inner_steps):
           a. Rebuild SFT-format batch (strip the R=0 rollout prefix so
              process_input_truncation can add a fresh-R prefix).
           b. Apply process_input_truncation with joint_masked_distribution —
              each k independently samples R and mask positions per sample.
           c. Forward: old_lp (frozen π_θ_old), ref_lp (frozen SFT ref),
                       cur_lp (current π_θ, with grad).
           d. GRPO loss (clipped surrogate + Schulman-k3 KL to ref); backward;
              optimizer.step. One optimizer update per inner step.

    accelerator.accumulate is NOT used around the inner loop: each inner step
    is an independent optimizer update, and we want a real DDP all-reduce per
    step so gradients reflect the per-mask loss exactly.
    """

    def __init__(self, model, optimizer, tokenizer, device,
                 train_dataloader, val_dataloader, test_dataloader,
                 use_fused, args):
        super().__init__(
            model, optimizer, tokenizer, device,
            train_dataloader, val_dataloader, test_dataloader,
            use_fused, args
        )

        self.grpo_group_size = getattr(args, 'grpo_group_size', 4)
        self.grpo_clip_eps = getattr(args, 'grpo_clip_eps', 0.2)
        self.grpo_kl_coeff = getattr(args, 'grpo_kl_coeff', 0.01)
        self.grpo_temperature = getattr(args, 'grpo_temperature', 0.7)
        self.grpo_top_p = getattr(args, 'grpo_top_p', 0.95)
        self.grpo_max_new_tokens = getattr(args, 'grpo_max_new_tokens', None) or self.args.max_new_tokens
        self.grpo_inner_steps = getattr(args, 'grpo_inner_steps', 1)

        # Built lazily after checkpoint load (see _train_process).
        self.ref_model = None
        self.old_model = None

    # ------------------------------------------------------------------
    # Frozen-model buffers
    # ------------------------------------------------------------------
    def _build_frozen_models(self):
        """Snapshot the current (possibly checkpoint-loaded) policy twice:
        - ref_model: KL-penalty reference, never updated.
        - old_model: π_θ_old buffer, refreshed from current policy at the
          start of every rollout via _sync_old_model.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)

        self.ref_model = copy.deepcopy(unwrapped)
        self.old_model = copy.deepcopy(unwrapped)

        for m in (self.ref_model, self.old_model):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            m.to(self.device)

    @torch.no_grad()
    def _sync_old_model(self):
        """Copy current policy weights into old_model (in-place, no new alloc).
        Called once per rollout, before the K-step inner loop."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.old_model.load_state_dict(unwrapped.state_dict(), strict=True)

    # ------------------------------------------------------------------
    # Rollout-prompt construction (R=0, reusing SFT string)
    # ------------------------------------------------------------------
    def _prepare_rollout(self, batch):
        """Returns (rollout_prompts, q_lengths, gt_answers).

        rollout_prompts: Q + "## masked 0 % of CoT ##" + EOS, one per sample.
          Reuses process_input_truncation(val_removal_p=0.0) so the prefix
          string is byte-identical to what SFT training produces at R=0.
        q_lengths: |Q| (without EOS), needed in _build_sft_batch to splice
          out the R=0 rollout prefix and let the scoring-side masking
          inject its own fresh-R prefix.
        gt_answers: extracted from the ORIGINAL (pre-processed) batch.
        """
        input_ids = batch['input_ids']
        first_sep_raw = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=0)

        r0_batch, _ = self.process_input_truncation(batch, val_removal_p=0.0)
        r0_ids = r0_batch['input_ids']
        first_sep_r0 = get_sep_position(r0_ids, self.tokenizer.eos_token_id, skip=0)

        rollout_prompts, q_lengths, gt_answers = [], [], []
        for i in range(input_ids.shape[0]):
            rollout_prompts.append(r0_ids[i, :first_sep_r0[i] + 1])
            q_lengths.append(first_sep_raw[i].item())

            tgt_text = self.tokenizer.decode(
                input_ids[i, first_sep_raw[i] + 1:], skip_special_tokens=True
            )
            gt_answers.append(extract_answer(tgt_text))

        return rollout_prompts, q_lengths, gt_answers

    # ------------------------------------------------------------------
    # Generation — routed through ImplicitModel.sample_completions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_completions(self, prompts):
        """Sample G completions per prompt (temperature sampling + DoubleEOS)."""
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        all_completions = []

        for prompt in prompts:
            with self.accelerator.autocast():
                outputs = unwrapped.sample_completions(
                    input_ids=prompt.unsqueeze(0),
                    max_new_tokens=self.grpo_max_new_tokens,
                    num_return_sequences=self.grpo_group_size,
                    temperature=self.grpo_temperature,
                    top_p=self.grpo_top_p,
                    stop_on_two_eos=True,
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
    # SFT batch from completions — strips the R=0 rollout prefix
    # ------------------------------------------------------------------
    def _is_valid_completion(self, completion):
        """A valid completion has >= 3 EOS tokens (rollout-prompt EOS, CoT EOS, answer EOS)."""
        return (completion == self.tokenizer.eos_token_id).sum().item() >= 3

    def _build_sft_batch(self, completions, q_lengths):
        """Rebuild plain SFT-format (Q + EOS + CoT_gen + EOS + A_gen + EOS).

        Input completion layout: Q + rollout_prefix + EOS + CoT_gen + EOS + A_gen + EOS
        The first EOS sits at the end of the rollout prefix, so comp[first_eos:]
        already gives us [EOS + CoT_gen + EOS + A_gen + EOS] — just prepend Q.
        """
        input_ids_list, labels_list = [], []
        for comp, q_len in zip(completions, q_lengths):
            eos_positions = (comp == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            first_eos = eos_positions[0].item()
            ids = torch.cat([comp[:q_len], comp[first_eos:]])
            lbl = ids.clone()
            lbl[:q_len + 1] = -100  # mask Q + first EOS
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
        mask[b,t] = 1 iff label[b,t+1] >= 0 (i.e. not masked out by -100)
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
        """Clipped surrogate objective + KL penalty.

        KL term is Schulman's k3 unbiased estimator of KL(π_θ ‖ π_ref):
          with log_r = log π_ref - log π_θ,  k3 = exp(log_r) - log_r - 1.
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio,
                              1 - self.grpo_clip_eps,
                              1 + self.grpo_clip_eps)

        adv = advantages.unsqueeze(-1)                       # [B, 1]
        policy_loss = -torch.min(ratio * adv, clipped * adv)

        log_r = ref_log_probs - log_probs
        kl = torch.exp(log_r) - log_r - 1                    # KL(π_θ ‖ π_ref)

        per_token = policy_loss + self.grpo_kl_coeff * kl
        per_sample = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)

        return per_sample.mean(), (kl * mask).sum() / mask.sum().clamp(min=1)

    # ------------------------------------------------------------------
    # Multi-GPU sync helper
    # ------------------------------------------------------------------
    def _any_rank(self, flag_bool):
        """True iff *any* rank has flag=True (used to sync skip decisions)."""
        t = torch.tensor([1 if flag_bool else 0], device=self.device, dtype=torch.long)
        gathered = self.accelerator.gather(t)
        return bool(gathered.sum().item() > 0)

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

        # Snapshot ref + initialise old buffer AFTER checkpoint load.
        self._build_frozen_models()

        if self.accelerator.is_main_process and self.writer:
            self.writer.set_step(step, mode="train")

        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process and self.writer:
                self.writer.set_step(step, mode="train")
                self.writer.add_scalar("epoch", epoch)

            if self.remove_by_schedule \
                    and self.schedule_index + 1 < len(self.masks_removal_schedule):
                if self.masks_removal_schedule[self.schedule_index + 1][0] == epoch:
                    self.schedule_index += 1
                    self.removal_p = self.masks_removal_schedule[self.schedule_index][1]

            loss_log = []
            print_line = f"Epoch {epoch}; Step {step}"
            if not self.joint_masked_distribution:
                print_line += f"; Scheduled to remove: {self.removal_p * 100}% of COT"
            if self.accelerator.is_main_process:
                print(print_line)
                print(f">>> Using mixed precision: {self.accelerator.state.mixed_precision}")

            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.train_dataloader),
                disable=not self.accelerator.is_main_process,
            ):
                # ---- rollout phase (no grad) ----
                self.model.eval()

                rollout_prompts, q_lengths, gt_answers = self._prepare_rollout(batch)
                completions = self._generate_completions(rollout_prompts)
                rewards = self._score_completions(completions, gt_answers)
                advantages = self._compute_advantages(rewards)

                all_rewards = [r for grp in rewards for r in grp]
                mean_reward = np.mean(all_rewards) if all_rewards else 0.0

                # flatten per-rank, tracking the source prompt's Q length
                comp_flat, adv_flat, q_len_flat = [], [], []
                for prompt_idx, (grp_c, grp_a) in enumerate(zip(completions, advantages)):
                    for c, a in zip(grp_c, grp_a):
                        if self._is_valid_completion(c):
                            comp_flat.append(c)
                            adv_flat.append(a)
                            q_len_flat.append(q_lengths[prompt_idx])

                has_local_signal = (
                    len(comp_flat) > 0
                    and torch.stack(adv_flat).abs().max().item() > 1e-8
                )

                # DDP-safe skip: if *no* rank has signal, all skip together.
                if not self._any_rank(has_local_signal):
                    continue

                # Ranks without local signal fabricate a 1-sample dummy so
                # DDP all-reduce stays in lockstep. adv=0 → policy loss 0;
                # only KL regularizer contributes (harmless).
                if not has_local_signal:
                    fallback_q_len = q_len_flat[0] if q_len_flat else q_lengths[0]
                    if not comp_flat:
                        comp_flat = [completions[0][0]]
                        q_len_flat = [fallback_q_len]
                    else:
                        comp_flat = [comp_flat[0]]
                        q_len_flat = [q_len_flat[0]]
                    adv_flat = [torch.zeros((), dtype=torch.float32)]

                adv_tensor = torch.stack(adv_flat).to(self.device)

                # ---- freeze π_θ_old for the inner loop ----
                self._sync_old_model()

                # ---- K inner GRPO updates, each with an independent mask pattern ----
                self.model.train()
                for k in range(self.grpo_inner_steps):
                    sft_batch = self._build_sft_batch(comp_flat, q_len_flat)
                    # joint_masked_distribution → random_masking samples a fresh
                    # R per sample AND fresh mask positions on every call.
                    truncated_batch, _ = self.process_input_truncation(sft_batch)

                    too_long = (
                        self.args.max_len_train > 0
                        and truncated_batch["input_ids"].shape[-1] > self.args.max_len_train
                    )
                    if self._any_rank(too_long):
                        if self.accelerator.is_main_process:
                            print("inner step skipped (too long on at least one rank)")
                        continue

                    with torch.no_grad():
                        with self.accelerator.autocast():
                            old_lp, _ = self._compute_per_token_log_probs(
                                self.old_model, truncated_batch)
                            ref_lp, _ = self._compute_per_token_log_probs(
                                self.ref_model, truncated_batch)

                    with self.accelerator.autocast():
                        cur_lp, mask = self._compute_per_token_log_probs(
                            self.model, truncated_batch)
                        loss, mean_kl = self._grpo_loss(
                            cur_lp, old_lp, ref_lp, mask, adv_tensor)

                    loss_log.append(loss.item())
                    self.accelerator.backward(loss)
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # ---- per-inner-step logging ----
                    if self.metrics_tracker is not None:
                        self.metrics_tracker.update("loss", loss.item())
                        self.metrics_tracker.update("grad_norm",
                                                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                        self.metrics_tracker.update("removal_p", self.removal_p)

                    if self.accelerator.is_main_process and step % 100 == 0:
                        print(f"Step: {step} (inner {k+1}/{self.grpo_inner_steps}). "
                              f"Loss: {loss.item():.4f}. "
                              f"Mean reward: {mean_reward:.3f}. "
                              f"KL: {mean_kl.item():.4f}")
                        if self.metrics_tracker is not None and self.writer is not None:
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
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch} avg loss: {avg:.4f}")

            eval_period = getattr(self.args, 'eval_period', 1)
            perform_gen_eval = (epoch % eval_period == 0) or \
                               (epoch == self.start_epoch + self.args.epochs - 1)

            save_best = self.evaluate(step, base_name="val",
                                      perform_generative_eval=perform_gen_eval)
            if self.args.test_path or \
                    (self.args.test_split and self.args.test_split != self.args.val_split):
                self.evaluate(step, base_name="test",
                              perform_generative_eval=perform_gen_eval)

            self.save_epoch(epoch, save_best=save_best)
