import math
import tqdm
import torch

from data import extract_answer
from utils import get_sep_position
from writer import WanDBWriter, MetricTracker

class BaseTrainer:
    def __init__(self, model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        self.args = args
        self.config = vars(args)

        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.epoch = 0
        self.writer = None
        self.metrics_tracker = None
        self.setup_metrics()

        self.device = device
        self.ptdtype = torch.bfloat16 if args.bf16 else torch.float32
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=self.ptdtype)
        self.use_fused = use_fused

    def setup_metrics(self):
        if self.args.wandb_project:
            self.writer = WanDBWriter(self.args)
            self.metrics_tracker = MetricTracker(
                *["loss", "perplexity", "accuracy", "token_accuracy", "grad_norm"],
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
            if self.writer: self.writer.log_tables()
            raise e
    
    def _train_process(self):
        pass

    def process_input_truncation(self):
        pass    

    @torch.no_grad()
    def evaluate(self, dataloader, name, truncation_kwargs, generation_kwargs):
        self.model.eval()
        self.metrics_tracker.reset()

        total_instances = 0
        total_tokens = 0
        total_correct = 0
        total_correct_tokens = 0
        total_loss = 0

        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
            input_ids_all = batch['input_ids'].to(self.device)
            labels_all = batch['labels'].to(self.device)

            input_ids_all, labels_all, position_ids_all, _ = self.process_input_truncation(batch, **truncation_kwargs)

            with self.ctx:
                if self.args.keep_position:
                    position_ids_all = position_ids_all[:, :input_ids_all.shape[-1]]
                outputs = self.model.compute_loss(input_ids=input_ids_all, labels=labels_all, position_ids=position_ids_all)

            total_loss += outputs.total_loss.item()
            total_correct_tokens += outputs.total_correct.item()
            total_tokens += outputs.total_tokens
            total_instances += input_ids_all.shape[0]

            # Generate + Evaluate
            # input_ids_all are cut to the start of COTs inside the model.generate
            if batch_idx % 100 == 0:
                first_sep_positions = get_sep_position(input_ids_all, self.tokenizer.eos_token_id)
                beam_outputs = self.model.generate(
                    input_ids=input_ids_all,
                    position_ids=position_ids_all,
                    **generation_kwargs
                )

                for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_outputs)):
                    tgt = input_ids_all_i[first_sep_positions[i] + 1:]
                    tgt_text = self.tokenizer.decode(tgt, skip_special_tokens=True)
                    ans = extract_answer(tgt_text)

                    pred_text = self.tokenizer.decode(beam_output_i[0][first_sep_positions[i] + 1:], skip_special_tokens=True)
                    pred_ans = extract_answer(pred_text)
                    if ans == pred_ans:
                        total_correct += 1
                    
                    query = self.tokenizer.decode(input_ids_all_i[:first_sep_positions[i]], skip_special_tokens=True)

                    print (f'Input: {query}')
                    print (f'Target: {tgt_text}')
                    print (f'Predicted: {pred_text}')
                    print ('')
                
        accuracy = total_correct / total_instances
        token_accuracy = total_correct_tokens / total_tokens
        loss = total_loss / total_tokens
        ppl = math.exp(loss)

        if self.metrics_tracker:
            self.metrics_tracker.update("loss", loss)
            self.metrics_tracker.update("perplexity", ppl)
            self.metrics_tracker.update("accuracy", accuracy)
            self.metrics_tracker.update("token_accuracy", token_accuracy)
            self.log_scalars(self.metrics_tracker)
            self.metrics_tracker.reset()

        print (f'Evaluation on part: {name}; PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        return accuracy, token_accuracy, ppl


    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        filename = str(self.args.save_model / "checkpoint-epoch{}.pth".format(epoch))

        if not (only_best and save_best):
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))
        
        if save_best:
            best_path = str(self.args.save_model / "model_best.pth")
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            print(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"] != self.config["optimizer"]:
            # checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
            print(
                "Warning: Optimizer given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
