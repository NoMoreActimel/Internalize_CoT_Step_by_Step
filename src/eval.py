import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import torch


from model import ImplicitModel
from configuration_model import ImplicitModelConfig
from data.data_chunked import add_new_tokens
from data.data import load_data
from trainers.trainer_stepbystep import StepByStepTrainer
from trainers.trainer_chunks import ChunkRemovalTrainer
from trainers.trainer_masks import AuxiliarMasksRemovalTrainer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def expand_gpt2_positions(model, args):
    old_length = model.base_model.transformer.wpe.weight.shape[0]
    if args.truncation > old_length and args.from_pretrained is None:
        print ('EXPANDING POSITIONS')
        new_wpe = torch.nn.Embedding(args.truncation, model.base_model.transformer.wpe.weight.shape[-1])
        new_wpe.weight.data[:old_length] = model.base_model.transformer.wpe.weight
        new_wpe.weight.data[old_length:] = model.base_model.transformer.wpe.weight[-1].view(1, -1).expand(args.truncation-old_length, -1)
        model.base_model.transformer.wpe = new_wpe

        for block in model.base_model.transformer.h:
            block.attn.register_buffer(
                "bias",
                torch.tril(torch.ones((args.truncation, args.truncation), dtype=torch.bool)).view(
                    1, 1, args.truncation, args.truncation
            ),
            persistent=False,
        )

def create_model(args, device):
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model)
        model = ImplicitModel(config, reinitialize_weights=args.train_from_scratch).to(device)
    else:
        print (f'Loading from {args.from_pretrained}')
        config = None
        model = ImplicitModel.from_pretrained(args.from_pretrained).to(device)

    if 'gpt2' in args.model:
        expand_gpt2_positions(model, args)
    
    model = model.to(device)
    tokenizer = model.tokenizer

    if args.reinitialize_weights:
        print ('reinitializing weights')
        model.model.apply(model.model._init_weights)

    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet
    
    return config, model, tokenizer


def parse_tuple_list(arg):
    try:
        pairs = [pair.strip("()").split(",") for pair in arg.split()]
        if "." in pairs[0][1]:
            return [(int(x), float(y)) for x, y in pairs]
        return [(int(x), int(y)) for x, y in pairs]
        # return [tuple(map(int, pair.strip("()").split(","))) for pair in arg.split()]
    except ValueError:
        raise argparse.ArgumentTypeError("Schedule must be in the format '(a,b)' with integers.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)

    parser.add_argument('--train_from_scratch', action='store_true')
    parser.set_defaults(train_from_scratch=False)

    parser.add_argument('--removal_type', type=str, choices=['step-by-step', 'random-chunks', 'random-masks'], default='step-by-step')

    # RANDOM MASKS REMOVAL
    parser.add_argument('--joint_masked_distribution', action='store_true')
    parser.set_defaults(joint_masked_distribution=False)
    parser.add_argument('--left_to_right_removal', action='store_true')
    parser.set_defaults(left_to_right_removal=False)
    # --remove_by_schedule and --removal_schedule from next section

    # RANDOM CHUNK REMOVAL
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--num_new_tokens', type=int, default=1000)


    parser.add_argument('--remove_chunks_step_by_step', action='store_true')
    parser.set_defaults(remove_chunks_step_by_step=False)

    parser.add_argument('--remove_by_schedule', action='store_true')
    parser.set_defaults(remove_by_schedule=False)

    # List of tuples: (from_epoch, n_chunks_removed), where n_chunks_removed == -1 -> remove all chunks
    default_schedule = "(0,0) (10,1) (30,2) (40,3) (50,-1)"
    parser.add_argument("--removal_schedule", type=parse_tuple_list, default=parse_tuple_list(default_schedule))

    parser.add_argument('--remove_when_flat_loss', action='store_true')
    parser.set_defaults(remove_when_flat_loss=False)
    parser.add_argument('--n_chunks_to_remove_from_start', type=int, default=0) # used with remove_when_flat_loss
    
    # ORIGINAL STEP-BY-STEP REMOVAL
    parser.add_argument('--remove_per_epoch', type=float, default=8)
    parser.add_argument('--remove_all_when_remove_beyond', type=str, default='inf')
    parser.add_argument('--removal_smoothing_lambda', type=float, default=float('inf'))
    parser.add_argument('--removal_side', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--max_size', type=int, default=-1)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--remove_start_from', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--reinitialize_weights', action='store_true')
    parser.set_defaults(reinitialize_weights=False)

    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    if args.remove_all_when_remove_beyond == 'inf':
        args.remove_all_when_remove_beyond = float('inf')
    else:
        args.remove_all_when_remove_beyond = int(args.remove_all_when_remove_beyond)

    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    # Create Model
    config, model, tokenizer = create_model(args, device)
    if args.removal_type == "random-chunks":
        model, tokenizer, new_token_ids = add_new_tokens(model, tokenizer, args.num_new_tokens + 1)
        start_id, new_token_ids = new_token_ids[0], new_token_ids[1:]
    else:
        new_token_ids = None

    # Load Data
    train_dataloader, val_dataloader, test_dataloader = load_data(args, tokenizer, new_token_ids)

    # Create Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)


    # Train
    if args.removal_type == "step-by-step":
        trainer = StepByStepTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
    elif args.removal_type == 'random-chunks':
        trainer = ChunkRemovalTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args, start_id)
    elif args.removal_type == 'random-masks':
        trainer =  AuxiliarMasksRemovalTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
    else:
        raise ValueError(f'args.removal_type must be either "step-by-step", "random-chunks" "random-masks", found {args.removal_type}')
    
    trainer.evaluate(
        trainer.val_dataloader,
        "val",
        trainer.val_truncation_kwargs,
        trainer.val_generation_kwargs,
        perform_generative_eval=True
    )

if __name__ == "__main__":
    main()

