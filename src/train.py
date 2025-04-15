import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import torch

from data import load_data
from data_chunked import add_new_tokens
from model import ImplicitModel
from model_jepa import JEPAImplicitModel
from configuration_model import ImplicitModelConfig
from trainer_stepbystep import StepByStepTrainer
from trainer_chunks import ChunkRemovalTrainer
from trainer_masks import AuxiliarMasksRemovalTrainer
from trainer_simple import SimpleTrainer


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

def create_model(args, ptdtype, device):
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model)
        model = ImplicitModel(
            config,
            reinitialize_weights=args.train_from_scratch,
            use_flash_attention=args.flash_attention_2
        ).to(device).to(ptdtype)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(
            args.from_pretrained,
            use_flash_attention=args.flash_attention_2
        ).to(device).to(ptdtype)
        config = model.config

    if 'gpt2' in args.model:
        expand_gpt2_positions(model, args)
    
    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer

    if args.reinitialize_weights:
        print ('reinitializing weights')
        model.model.apply(model.model._init_weights)

    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet
    
    return config, model, tokenizer

def create_jepa_model(config, ref_model, args, ptdtype, device):
    if args.from_pretrained is None:
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=args.alpha_logits_loss,
            ref_model_update_decay=args.ref_model_update_decay,
            logits_loss_on_full_cot=args.logits_loss_on_full_cot,
            reinitialize_weights=args.train_from_scratch,
            use_flash_attention=args.flash_attention_2
        ).to(device).to(ptdtype)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=args.alpha_logits_loss,
            ref_model_update_decay=args.ref_model_update_decay,
            logits_loss_on_full_cot=args.logits_loss_on_full_cot,
            reinitialize_weights=False,
            use_flash_attention=args.flash_attention_2
        ).to(device).to(ptdtype)
        state_dict = torch.load(os.path.join(args.from_pretrained, 'state_dict.bin'))
        model.load_state_dict(state_dict, strict=True)

    if 'gpt2' in args.model:
        expand_gpt2_positions(model, args)
    
    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet
    
    return config, model, model.tokenizer


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
    parser.add_argument('--train_path', type=str, required=False)
    parser.add_argument('--val_path', type=str, required=False)
    parser.add_argument('--test_path', type=str, default=None)

    # HuggingFace Dataset
    parser.add_argument('--huggingface_dataset', action='store_true', default=False)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--data_files', type=str, default=None)
    parser.add_argument('--max_samples', type=str, default=None)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--shuffle_seed', type=int, default=None)
    parser.add_argument('--question_key', type=str, default="question")
    parser.add_argument('--answer_key', type=str, default="answer")

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)

    parser.add_argument('--n_generative_eval_batches', type=int, default=1)

    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')

    parser.add_argument('--removal_type', type=str, choices=['step-by-step', 'random-chunks', 'random-masks'], default='step-by-step')

    parser.add_argument('--train_type', type=str, choices=['full-cot', 'cot-distill', 'jepa-cot-distill'], default='cot-distill', help="""
                        Type of Training:
                        - 'full-cot': simply train on full Chains-of-Thought data
                        - 'cot-distill': for different types of Chains-of-Thought distillation, by removal_type (default)
                        - 'jepa-cot-distill': for JEPA-like COT distillation training.
                            That is, match logits / last hidden states on [answer] part
                            between reference model on full COTs and current model on corrupted COTs.
                            Works only with --removal_type 'random-masks' trainer.
                        """)

    # JEPA-like training flags    
    parser.add_argument('--alpha_logits_loss', type=float, default=1.0, help='JEPA loss multiplier')

    parser.add_argument('--ref_model_update_decay', type=float, default=0.999, help="""
                        Coefficient for EMA on ref model weights in JEPA training""")
    
    parser.add_argument('--logits_loss_on_full_cot', action='store_true', default=False, help="""
                        Whether to put logits loss on full-COT + answer, or answer only (by default)""")

    # RANDOM MASKS REMOVAL

    # Sample N_tokens_removed for each sample independently
    parser.add_argument('--joint_masked_distribution', action='store_true', default=False)

    # Select random tokens in mask to remove (default if removal_type = 'random_masks')

    # Select contiguous mask of random length from left-to-right:
    parser.add_argument('--left_to_right_removal', action='store_true', default=False)
    # Select contiguous mask of random length starting from random position 
    parser.add_argument('--random_contiguous_removal', action='store_true', default=False)

    # Replace the masked tokens with special mask_id (removed by default):
    parser.add_argument('--replace_mask', action='store_true', default=False)
    parser.add_argument('--replace_mask_in_labels', action='store_true', default=False, help="""
                        Target labels on training will have masked COTs.
                        Works only together with --replace_mask""")

    # if we need to remove ## removed N ## hint - fine-tune on plain no-COT data
    parser.add_argument('--no_cot_stage', action='store_true', default=False)

    # --remove_by_schedule and --removal_schedule from next section

    # RANDOM CHUNK REMOVAL
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--num_new_tokens', type=int, default=1000)

    parser.add_argument('--remove_chunks_step_by_step', action='store_true', default=False)
    parser.add_argument('--remove_by_schedule', action='store_true', default=False)

    # List of tuples: (from_epoch, n_chunks_removed), where n_chunks_removed == -1 -> remove all chunks
    default_schedule = "(0,0) (10,1) (30,2) (40,3) (50,-1)"
    parser.add_argument("--removal_schedule", type=parse_tuple_list, default=parse_tuple_list(default_schedule))

    parser.add_argument('--remove_when_flat_loss', action='store_true', default=False)
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
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--from_pretrained', type=str, default=None, help="Load model weights only")
    parser.add_argument('--from_pretrained_checkpoint', type=str, default=None, help="Load model weights and optimizer state")

    # TRAIN JEPA FROM FULL-COT MODEL
    parser.add_argument('--from_pretrained_fullcot_checkpoint', type=str, default=None, help="Load ref model weights from full-COT checkpoint")

    parser.add_argument('--remove_start_from', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--flash_attention_2', action='store_true', default=False)
    # pip install flash-attn --no-build-isolation

    parser.add_argument('--reset_optimizer', action='store_true', default=False)
    parser.add_argument('--keep_position', action='store_true', default=False)
    parser.add_argument('--reinitialize_weights', action='store_true', default=False)

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

    dtype = 'bfloat16' if args.bf16 else 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(ptdtype, dtype, device)

    # Create Model
    config, model, tokenizer = create_model(args, ptdtype, device)
    if args.removal_type == "random-chunks":
        model, tokenizer, new_token_ids = add_new_tokens(model, tokenizer, args.num_new_tokens + 1)
        start_id, new_token_ids = new_token_ids[0], new_token_ids[1:]
    else:
        new_token_ids = None

    if args.train_type == "jepa-cot-distill":
        ref_model = model
        _, model, __ = create_jepa_model(config, ref_model, args, ptdtype, device)

    # Load Data
    train_dataloader, val_dataloader, test_dataloader = load_data(args, tokenizer, new_token_ids)

    # Create Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)


    # Train
    if args.train_type == "full-cot":
        trainer = SimpleTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
    elif args.removal_type == "step-by-step":
        trainer = StepByStepTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
    elif args.removal_type == 'random-chunks':
        trainer = ChunkRemovalTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args, start_id)
    elif args.removal_type == 'random-masks':
        trainer =  AuxiliarMasksRemovalTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
    else:
        raise ValueError(f'args.removal_type must be either "step-by-step", "random-chunks" "random-masks", found {args.removal_type}')
    
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        trainer.evaluate(
            trainer.val_dataloader,
            "val",
            trainer.val_truncation_kwargs,
            trainer.val_generation_kwargs,
            perform_generative_eval=True
        )
    else:
        raise ValueError(f'args.mode must be either "train" or "eval", found {args.mode}')



if __name__ == "__main__":
    main()
