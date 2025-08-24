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
from data_distillation import load_distillation_data
from model import ImplicitModel
from model_jepa import JEPAImplicitModel
from configuration_model import ImplicitModelConfig
from trainer_stepbystep import StepByStepTrainer
from trainer_chunks import ChunkRemovalTrainer
from trainer_masks import AuxiliarMasksRemovalTrainer
from trainer_simple import SimpleTrainer
from trainer_distillation import DistillationTrainer


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
        config = ImplicitModelConfig(base_model=args.model, n_head=args.n_head)
        model = ImplicitModel(
            config,
            reinitialize_weights=args.train_from_scratch,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(
            args.from_pretrained,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
        config = model.config

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

def create_jepa_model(config, ref_model, args, device):
    if args.from_pretrained is None:
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=args.alpha_logits_loss,
            ref_model_update_decay=args.ref_model_update_decay,
            logits_loss_on_full_cot=args.logits_loss_on_full_cot,
            reinitialize_weights=args.train_from_scratch,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=args.alpha_logits_loss,
            ref_model_update_decay=args.ref_model_update_decay,
            logits_loss_on_full_cot=args.logits_loss_on_full_cot,
            reinitialize_weights=False,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
        state_dict = torch.load(os.path.join(args.from_pretrained, 'state_dict.bin'))
        model.load_state_dict(state_dict, strict=True)

    if 'gpt2' in args.model:
        expand_gpt2_positions(model, args)
    
    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet
    
    return config, model, model.tokenizer


def add_jump_tokens(model, tokenizer, num_jump_tokens):
    orig_vocab_length = len(tokenizer.get_vocab())

    new_tokens = [f"<JUMP_{i+1}>" for i in range(num_jump_tokens)]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    vocab_length = len(tokenizer.get_vocab())

    assert num_added_tokens == num_jump_tokens, "Jump tokens addition failed!"
    assert orig_vocab_length + num_jump_tokens == vocab_length, "Jump tokens addition failed!"

    jump_token_ids = [id for id in range(orig_vocab_length, vocab_length)]
    model.base_model.resize_token_embeddings(len(tokenizer))
    model.jump_token_ids = jump_token_ids
    return model, tokenizer


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
    parser.add_argument('--n_head', type=int, default=None)

    parser.add_argument('--train_path', type=str, required=False)
    parser.add_argument('--val_path', type=str, required=False)
    parser.add_argument('--test_path', type=str, default=None)

    # HuggingFace Dataset
    parser.add_argument('--huggingface_dataset', action='store_true', default=False)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--train_split', type=str, default="train")
    parser.add_argument('--val_split', type=str, default="val")
    parser.add_argument('--test_split', type=str, default=None)
    parser.add_argument(
        '--json_dataset', action='store_true',
        help="Whether dataset is stored in .json (list of dicts with question/answer fields), default is row-wise .txt"
    )
    parser.add_argument('--data_files', type=str, default=None)
    parser.add_argument('--max_samples', type=str, default=None)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--shuffle_seed', type=int, default=None)
    parser.add_argument('--question_key', type=str, default="question")
    parser.add_argument('--answer_key', type=str, default="answer")

    parser.add_argument('--pad_cot', action='store_true', default=False)
    parser.add_argument('--max_cot_length', type=int, default=128)
    parser.add_argument('--cot_pad_id', type=int, default=None)
    parser.add_argument('--pad_query', action='store_true', default=False)
    parser.add_argument('--max_query_length', type=int, default=128)
    parser.add_argument('--query_pad_id', type=int, default=None)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)

    # Distillation hyperparameters
    parser.add_argument('--distillation', action='store_true', default=False,
                        help='Enable distillation training mode using teacher generations and logits')
    parser.add_argument('--distillation_temperature', type=float, default=4.0,
                        help='Temperature for distillation softmax')
    parser.add_argument('--distillation_alpha', type=float, default=0.7,
                        help='Weight for KL divergence loss')
    parser.add_argument('--distillation_beta', type=float, default=0.3,
                        help='Weight for cross-entropy loss')

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
    
    # Training with random CoTs
    parser.add_argument('--random_cot', action='store_true', default=False, help="""Whether to use random CoTs""")
    parser.add_argument('--random_cot_length', type=int, default=50, help="""Length of random CoT to generate.""")
    parser.add_argument(
        '--random_cot_strategy', type=str,
        choices=['random_digits', 'random_digits_from_question', 'fixed_permutation_random_digits_from_question'],
        default='random_digits', help="""
            Strategy for generating random CoT:
            - 'random_digits': Completely random digits separated with space
            - 'random_digits_from_question': Randomly sampled tokens from question (digits), new permutation each sample
            - 'fixed_permutation_random_digits_from_question': Same as 2, but fixed permutation for all samples,
                All samples MUST have the same number of digits in question.
        """
    )

    # Validation metric name by which to save the best model
    parser.add_argument(
        '--best_val_metric', type=str,
        choices=['accuracy', 'token-accuracy', 'full-cot_accuracy', 'no-cot_accuracy', 'full-cot_token-accuracy', 'no-cot_token-accuracy'],
        default='accuracy'
    )
    parser.add_argument(
        '--parallel_cot_inference_on_full_removal', action='store_true',
        help='Whether to use one forward pass on inference in full-masked mode, works only with random_masks on val_removal_p = 1.0'
    )

    # JEPA-like training flags    
    parser.add_argument('--alpha_logits_loss', type=float, default=1.0, help='JEPA loss multiplier')

    parser.add_argument('--ref_model_update_decay', type=float, default=0.999, help="""
                        Coefficient for EMA on ref model weights in JEPA training""")
    
    parser.add_argument('--logits_loss_on_full_cot', action='store_true', default=False, help="""
                        Whether to put logits loss on full-COT + answer, or answer only (by default)""")

    # RANDOM MASKS REMOVAL

    # Sample N_tokens_removed for each sample independently
    parser.add_argument('--joint_masked_distribution', action='store_true', default=False)

    parser.add_argument('--mask_beta_sampling', action='store_true', default=False, help="""
        Sample mask length according to Beta(a, a) instead of U[0, 1]
                        
        This has two effects:
        1. Masks of lengths close to 0 and to N will be selected much more frequently if a -> 0.
           That is, model will be trained on edge-cases more frequently in any joint masking / removal regime
        2. For contiguous regimes (random contiguous, left-to-right) it rebalances the probability of maksing tokens in the middle of CoT.
           When length is sampled uniformly, middle of CoT is getting masked more frequently than its sides.
           To balance position masking probabilities, we sample length from Beta(a, a). When a -> 0, the distribution becomes more equal.
    """)
    parser.add_argument('--mask_beta_sampling_param', type=float, default=0.1)

    # Select random tokens in mask to remove (default if removal_type = 'random_masks')

    # Select contiguous mask of random length from left-to-right:
    parser.add_argument('--left_to_right_removal', action='store_true', default=False)
    # Select contiguous mask of random length starting from random position 
    parser.add_argument('--random_contiguous_removal', action='store_true', default=False)
    parser.add_argument('--use_jump_tokens', action='store_true', default=False, help="""
        Instead of new prompt, use special <JUMP_i> tokens inside chain-of-thoughts.
    """)
    parser.add_argument('--num_jump_tokens', type=int, default=256, help="""
        Number of jump tokens to use, must be >= max CoT length.
    """)

    # Replace the masked tokens with special mask_id (removed by default):
    parser.add_argument('--replace_mask', action='store_true', default=False, help="""
                        Instead of cutting selected CoT tokens, replace them with a MASK token.
                        """)
    parser.add_argument('--replace_mask_in_labels', action='store_true', default=False, help="""
                        Target labels on training will have masked COTs.
                        Otherise, in case of replace_mask = True, labels will consist of all CoT
                        Works only together with --replace_mask""")
    
    # will use contiguous masks on eval even for random masks method
    # this will use model.generate calls instead of for on forwards, with kv-cache
    parser.add_argument('--eval_on_contiguous_masks', action='store_true', default=False)

    parser.add_argument('--intermediate_eval', action='store_true', default=False)

    # Whether to put ## masked N ## hint or ## masked R% ## instead
    # This one is better but harder, as it exludes cheating on validation
    parser.add_argument('--prompt_in_percentage', action='store_true', default=False)

    # if we need to remove ## masked N ## hint - fine-tune on plain no-COT data
    parser.add_argument('--no_cot_stage', action='store_true', default=False)
    parser.add_argument('--no_cot_stage_mask_length', type=int, default=None)

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
    parser.add_argument('--train_max_size', type=int, default=None)
    parser.add_argument('--val_max_size', type=int, default=None)
    parser.add_argument('--test_max_size', type=int, default=None)

    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--from_pretrained', type=str, default=None, help="Load model weights only")
    parser.add_argument('--from_pretrained_checkpoint', type=str, default=None, help="Load model weights and optimizer state")
    parser.add_argument('--resume_without_optimizer', action='store_true', default=False)

    # TRAIN JEPA FROM FULL-COT MODEL, EQUIVALENT TO FROM_PRETRAINED_CHECKPOINT + RESUME_WITHOUT_OPTIMIZER, LEGACY:
    parser.add_argument('--from_pretrained_fullcot_checkpoint', type=str, default=None, help="Load ref model weights from full-COT checkpoint")

    parser.add_argument('--remove_start_from', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # does not work with jepa training, uses default lora config in src/model.py
    parser.add_argument('--use_peft', action='store_true', default=False)

    # pip install flash-attn --no-build-isolation
    parser.add_argument('--flash_attention_2', action='store_true', default=False)

    # only for step-by-step removal:
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create Model
    config, model, tokenizer = create_model(args, device)
    if args.removal_type == "random-chunks":
        model, tokenizer, new_token_ids = add_new_tokens(model, tokenizer, args.num_new_tokens + 1)
        start_id, new_token_ids = new_token_ids[0], new_token_ids[1:]
    else:
        new_token_ids = None

    if args.train_type == "jepa-cot-distill":
        ref_model = model
        _, model, __ = create_jepa_model(config, ref_model, args, device)
    
    print("MODEL ARCHITECTURE:")
    print(model)
    
    if args.model == "gpt2" and args.n_head is not None:
        for idx, layer in enumerate(model.base_model.transformer.h):
            if hasattr(layer, "attn"):
                print(f"Layer {idx + 1}, num attn heads: {layer.attn.num_heads}")

    print("MODEL ARCHITECTURE:")
    print(model)
    
    if args.model == "gpt2" and args.n_head is not None:
        for idx, layer in enumerate(model.base_model.transformer.h):
            if hasattr(layer, "attn"):
                print(f"Layer {idx + 1}, num attn heads: {layer.attn.num_heads}")

    if args.use_jump_tokens:
        model, tokenizer = add_jump_tokens(model, tokenizer, num_jump_tokens=args.num_jump_tokens)

    # Load Data
    if args.distillation:
        print("Loading distillation data...")
        train_dataloader, val_dataloader, test_dataloader = load_distillation_data(
            train_dir_path=args.train_path,
            val_dir_path=args.val_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_len_train if args.max_len_train > 0 else None,
            test_dir_path=args.test_path
        )
    else:
        train_dataloader, val_dataloader, test_dataloader = load_data(args, tokenizer, new_token_ids)

    # Create Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    # Train
    if args.distillation:
        trainer = DistillationTrainer(model, optimizer, tokenizer, device, train_dataloader, val_dataloader, test_dataloader, use_fused, args)
    elif args.train_type == "full-cot":
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
        step = 0
        if trainer.args.from_pretrained_checkpoint:
            step = trainer._resume_checkpoint(
                trainer.args.from_pretrained_checkpoint,
                load_optimizer_state=not trainer.args.resume_without_optimizer
            )
        trainer.evaluate(step=step)
    else:
        raise ValueError(f'args.mode must be either "train" or "eval", found {args.mode}')



if __name__ == "__main__":
    main()
