import torch


class BaseTrainer:
    def __init__(self, model, optimizer, tokenizer, train_dataloader, val_dataloader, test_dataloader, use_fused, args):
        self.model = model
        self.trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.device = self.model.device
        self.ptdtype = torch.bfloat16 if args.bf16 else torch.float32
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=self.ptdtype)
        self.use_fused = use_fused

        self.args = args
    
    def train(self):
        pass

    def evaluate(self):
        pass
