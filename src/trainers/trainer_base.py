import math
import tqdm
import torch

from accelerate import Accelerator
from transformers.utils import is_peft_available

from ..writer import WanDBWriter, MetricTracker
from ..utils import get_sep_position, extract_answer

if is_peft_available():
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

class BaseTrainer:
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        self.args = args
        self.config = vars(args)

        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.accumulate)
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            self.accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_dataloader)
        self.tokenizer = tokenizer

        mp = self.accelerator.state.mixed_precision
        dtype = torch.bfloat16 if mp == "bf16" else (torch.float16 if mp == "fp16" else torch.float32)
        self.model = self.model.to(dtype)

        print(f"USING DTYPE {dtype} FOR TRAINING!")

        self.device = self.accelerator.device
        self.use_fused = use_fused
        self.jepa_training = hasattr(self.model, "ref_model")

        self.epoch = 0
        self.start_epoch = 0
        self.writer = None
        self.metrics_tracker = None

        self.best_val_metric_type = self.args.best_val_metric
        self.best_val_metric = 0.0

    def check_best(self, acc, tok_acc, ppl):
        save_best = False
        if self.best_val_metric_type == "accuracy":
            if acc > self.best_val_metric:
                self.best_val_metric = acc
                save_best = True
        elif self.best_val_metric_type == "token-accuracy":
            if tok_acc > self.best_val_metric:
                self.best_val_metric = tok_acc
                save_best = True
        elif self.best_val_metric_type == "perplexity":
            if ppl > self.best_val_metric:
                self.best_val_metric = ppl
                save_best = True
        return save_best

    def setup_metrics(self, additional_metrics=None):
        if self.args.wandb_project:
            self.writer = WanDBWriter(self.args)
            if additional_metrics is None:
                additional_metrics = []
            self.metrics_tracker = MetricTracker(
                *["loss", "perplexity", "accuracy", "token_accuracy", "grad_norm"],
                *additional_metrics,
                writer=self.writer
            )
    
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    @staticmethod
    def move_batch_to_device(batch, device):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
    
    def _reset_optimizer(self):
        print ('RESETTING OPTIMIZER')
        self.optimizer.zero_grad(set_to_none=True)
        del self.optimizer
        self.optimizer = torch.optim.AdamW(self.get_trainable_params(), lr=self.args.lr, **{"fused": self.use_fused})
    
    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            print("Saving model on keyboard interrupt")
            self._save_checkpoint(self.epoch, save_best=False)
            if self.accelerator.is_main_process and self.writer:
                self.writer.log_tables()
            raise e
    
    def _train_process(self):
        pass

    def process_input_truncation(self):
        pass

    def evaluate(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)

    @torch.no_grad()
    def _evaluate(
            self,
            dataloader,
            name,
            truncation_kwargs,
            generation_kwargs,
            perform_generative_eval=True,
            generative_eval_hooks=[],
            generative_eval_single_batch_size=False
    ):
        self.model.eval()
        self.metrics_tracker.reset()

        total_tokens = 0
        total_correct = 0
        total_correct_tokens = 0
        total_loss = 0
        total_generated = 0

        for batch_idx, batch_initial in tqdm.tqdm(enumerate(dataloader)):
            batch, _ = self.process_input_truncation(batch_initial, **truncation_kwargs)
            # self.move_batch_to_device(batch, self.device) <-- handled by accelerator

            with self.accelerator.autocast():
                if self.args.keep_position:
                    batch["position_ids"] = batch["position_ids"][:, :batch["input_ids"].shape[-1]]
                outputs = self.model.compute_loss(**batch)

            total_loss += outputs.total_loss.item()
            total_correct_tokens += outputs.total_correct.item()
            total_tokens += outputs.total_tokens

        # separate gen eval for batch_size = 1 in case of mask insertion
        if perform_generative_eval:
            for batch_idx, batch_initial in tqdm.tqdm(enumerate(dataloader)):
                if batch_idx >=  getattr(self.args, "n_generative_eval_batches", 1):
                    break
                
                batch, _ = self.process_input_truncation(batch_initial, **truncation_kwargs)
                with self.accelerator.autocast():
                    if self.args.keep_position:
                        batch["position_ids"] = batch["position_ids"][:, :batch["input_ids"].shape[-1]]
                
                if generative_eval_single_batch_size and batch["input_ids"].shape[0] != 1:
                    # all batch values should have batch_size = 1 to support non-equal CoT sizes
                    batches = []
                    for i in range(batch["input_ids"].shape[0]):
                        batch_i = {}
                        for k, v in batch.items():
                            batch_i[k] = v
                            if isinstance(v, torch.Tensor):
                                batch_i[k] = v[i].unsqueeze(0)
                        batches.append(batch_i)
                else:
                    batches = [batch]
                            
                print(f'\nGENERATE BATCH_IDX: {batch_idx} / {getattr(self.args, "n_generative_eval_batches", 1)}')
        
                for batch in batches:
                    # Generate + Evaluate
                    # input_ids_all are cut to the start of COTs inside the model.generate
                    for hook in generative_eval_hooks:
                        generation_kwargs = hook(
                            batch=batch,
                            truncation_kwargs=truncation_kwargs,
                            generation_kwargs=generation_kwargs
                        )
                        
                    first_sep_positions = get_sep_position(batch["input_ids"], self.tokenizer.eos_token_id)
                    with self.accelerator.autocast():
                        beam_outputs = self.model.generate(
                            input_ids=batch["input_ids"],
                            position_ids=batch["position_ids"],
                            **generation_kwargs
                        )

                    total_generated += batch["input_ids"].shape[0]

                    for i, (input_ids_all_i, beam_output_i) in enumerate(zip(batch["input_ids"], beam_outputs)):
                        tgt = input_ids_all_i[first_sep_positions[i] + 1:]
                        tgt_text = self.tokenizer.decode(tgt, skip_special_tokens=True)
                        ans = extract_answer(tgt_text)

                        pred_text = self.tokenizer.decode(beam_output_i[0][first_sep_positions[i] + 1:], skip_special_tokens=True)
                        pred_ans = extract_answer(pred_text)
                        if ans == pred_ans:
                            total_correct += 1
                        
                        query = self.tokenizer.decode(input_ids_all_i[:first_sep_positions[i]], skip_special_tokens=True)
                        if i <= 3:
                            print (f'Input: {query}')
                            print (f'Target: {tgt_text}')
                            print (f'Predicted: {pred_text}')
                            print ('')
        
        reduce_func = lambda x: self.accelerator.reduce(torch.tensor(x, device=self.device), reduction="sum").item()
        total_loss = reduce_func(total_loss)
        total_correct_tokens = reduce_func(total_correct_tokens)
        total_tokens = reduce_func(total_tokens)
        total_generated = reduce_func(total_generated)
        total_correct = reduce_func(total_correct)

        accuracy = total_correct / total_generated if perform_generative_eval else 0.0   
        token_accuracy = total_correct_tokens / total_tokens
        loss = total_loss / total_tokens
        ppl = math.exp(loss)

        if self.accelerator.is_main_process and self.metrics_tracker:
            self.metrics_tracker.update("loss", loss)
            self.metrics_tracker.update("perplexity", ppl)
            self.metrics_tracker.update("accuracy", accuracy)
            self.metrics_tracker.update("token_accuracy", token_accuracy)

            if self.jepa_training:
                self.metrics_tracker.update("CE_loss", outputs.ce_loss.item())
                self.metrics_tracker.update("JEPA_loss", outputs.logits_loss.item())
            
            self.log_scalars(self.metrics_tracker)
            self.metrics_tracker.reset()

        print (f'Evaluation on part: {name}; PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        return accuracy, token_accuracy, ppl

    def save_epoch(self, epoch, save_best=False):
        if epoch % self.args.save_period == 0 or epoch == self.args.epochs - 1:
            # self.model.save_pretrained(os.path.join(self.args.save_model, f'checkpoint_{epoch}'))
            self._save_checkpoint(epoch, save_best=save_best, only_best=False)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        if not self.accelerator.is_main_process:
            return

        arch = type(self.model).__name__

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        state = {
            "arch": arch,
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        if getattr(unwrapped_model, "use_peft", False):
            assert is_peft_available()
            state["state_dict"] = get_peft_model_state_dict(unwrapped_model.base_model)
            # state["peft_config"] = unwrapped_model.peft_config
        else:
            state["state_dict"] = unwrapped_model.state_dict()
            if hasattr(self.model, "ref_model"):
                state["ref_state_dict"] = unwrapped_model.ref_model.state_dict()

        if getattr(unwrapped_model, "use_peft", False):
            filename = str(self.args.save_model + "/checkpoint-epoch{}-adapter.pth".format(epoch))
        else:
            filename = str(self.args.save_model + "/checkpoint-epoch{}.pth".format(epoch))

        if not (only_best and save_best):
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))
        
        if save_best:
            best_path = str(self.args.save_model + "/model_best.pth")
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, load_optimizer_state=True):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))

        with self.accelerator.main_process_first():
            checkpoint = torch.load(resume_path, map_location=self.device)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        self.start_epoch = checkpoint["epoch"] + 1
        step = self.start_epoch * ((len(self.train_dataloader) + self.args.batch_size - 1) // self.args.batch_size)

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            print(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        
        if getattr(unwrapped_model, "use_peft", False):
            # unwrapped_model.set_peft_config(checkpoint["peft_config"])
            unwrapped_model.become_peft_model()
            set_peft_model_state_dict(unwrapped_model, checkpoint["state_dict"])
        else:
            if not getattr(self, "jepa_training", False):
                unwrapped_model.load_state_dict(checkpoint["state_dict"])
            else:
                unwrapped_model.load_state_dict(checkpoint["state_dict"], strict=False)
                if "ref_state_dict" in checkpoint:
                    unwrapped_model.ref_model.load_state_dict(checkpoint["ref_state_dict"])
                    print("Loaded ref_model from the previous ref_state_dict!")
                else:
                    unwrapped_model.ref_model.load_state_dict(checkpoint["state_dict"])
                    print("Loaded ref_model from the default model's state_dict!")
                unwrapped_model.ref_model.eval()
            
        if load_optimizer_state:
            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint["config"].get("optimizer", "") != self.config.get("optimizer", ""):
                # checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
                print(
                    "Warning: Optimizer given in config file is different "
                    "from that of checkpoint. Optimizer parameters not being resumed."
                )
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            print("Only Model weights were loaded, Optimizer state was not. Resume training from epoch {}".format(self.start_epoch))

        if self.accelerator.is_main_process:
            print(f"Loaded checkpoint '{resume_path}'. Resume training from epoch {self.start_epoch}")

        self.accelerator.wait_for_everyone()
        return step

    def log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
