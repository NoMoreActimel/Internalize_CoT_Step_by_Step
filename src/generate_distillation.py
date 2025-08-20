import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import numpy as np
import torch
import pickle

from data import load_data
from data_chunked import add_new_tokens
from model import ImplicitModel
from model_jepa import JEPAImplicitModel
from configuration_model import ImplicitModelConfig
from utils import get_sep_position, extract_answer, extract_cot
from train import create_model, expand_gpt2_positions

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file (.pth)"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_from_checkpoint = checkpoint["config"]
    
    # Create model config
    config = ImplicitModelConfig(
        base_model=config_from_checkpoint["model"], 
        n_head=config_from_checkpoint.get("n_head", None)
    )
    
    # Determine if it's a JEPA model
    is_jepa = "ref_state_dict" in checkpoint or config_from_checkpoint.get("train_type") == "jepa-cot-distill"
    
    if is_jepa:
        # Create a temporary reference model first
        ref_model = ImplicitModel(
            config,
            reinitialize_weights=False,
            use_flash_attention=config_from_checkpoint.get("flash_attention_2", False),
            use_peft=config_from_checkpoint.get("use_peft", False)
        ).to(device)
        
        # Create JEPA model
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=config_from_checkpoint.get("alpha_logits_loss", 1.0),
            ref_model_update_decay=config_from_checkpoint.get("ref_model_update_decay", 0.999),
            logits_loss_on_full_cot=config_from_checkpoint.get("logits_loss_on_full_cot", False),
            reinitialize_weights=False,
            use_flash_attention=config_from_checkpoint.get("flash_attention_2", False),
            use_peft=config_from_checkpoint.get("use_peft", False)
        ).to(device)
    else:
        # Create regular model
        model = ImplicitModel(
            config,
            reinitialize_weights=False,
            use_flash_attention=config_from_checkpoint.get("flash_attention_2", False),
            use_peft=config_from_checkpoint.get("use_peft", False)
        ).to(device)
    
    # Handle GPT2 position expansion if needed
    if 'gpt2' in config_from_checkpoint["model"]:
        truncation = config_from_checkpoint.get("truncation", -1)
        if truncation > 0:
            # Create a temporary args object for expand_gpt2_positions
            class TempArgs:
                def __init__(self, truncation):
                    self.truncation = truncation
                    self.from_pretrained = None
            
            temp_args = TempArgs(truncation)
            expand_gpt2_positions(model, temp_args)
    
    tokenizer = model.tokenizer
    
    # Load model state
    try:
        if hasattr(model, "use_peft") and model.use_peft:
            try:
                from peft import set_peft_model_state_dict
                model.become_peft_model()
                set_peft_model_state_dict(model, checkpoint["state_dict"])
            except ImportError:
                raise ImportError("PEFT library not installed. Install with: pip install peft")
        else:
            if is_jepa:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                if "ref_state_dict" in checkpoint:
                    model.ref_model.load_state_dict(checkpoint["ref_state_dict"])
                    print("Loaded ref_model from checkpoint ref_state_dict!")
                else:
                    model.ref_model.load_state_dict(checkpoint["state_dict"])
                    print("Loaded ref_model from main state_dict!")
                model.ref_model.eval()
            else:
                model.load_state_dict(checkpoint["state_dict"])
        
        print(f"Successfully loaded model from checkpoint at epoch {checkpoint['epoch']}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    return config, model, tokenizer


def _get_sep_positions(input_ids, tokenizer):
    """Get positions of first, second EOS tokens and final EOS"""
    first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
    second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
    eos_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)
    return first_sep_positions, second_sep_positions, eos_positions


def _get_prefix_contiguous(tokenizer, n_tokens_to_remove, removal_p, device, left_to_right=True, L=0, prompt_in_percentage=False):
    """Copy of _get_prefix_contiguous_truncation from trainer_masks.py"""
    if prompt_in_percentage:
        prefix_str = f" ## masked {round(removal_p * 100)} % of CoT ## "
    else:
        if left_to_right:
            prefix_str = f" ## masked {n_tokens_to_remove} ## "
        else:
            prefix_str = f" ## masked {n_tokens_to_remove} from {L} ## "

    prefix = tokenizer(
        prefix_str,
        add_special_tokens=True,
        truncation=True,
        return_tensors="pt"
    )["input_ids"].to(device).squeeze(0)

    eos = torch.full((1,), tokenizer.eos_token_id, dtype=torch.long).to(device)
    prefix = torch.cat([prefix, eos])

    ignored_prefix_labels = torch.full_like(prefix, -100)
    return prefix, ignored_prefix_labels


def _get_prefix_random_masking(tokenizer, cot_start, removed_indices, removal_p, device, prompt_in_percentage=False):
    """Copy of _get_prefix_random_masking from trainer_masks.py"""
    if prompt_in_percentage:
        prefix_str = f" ## masked {round(removal_p * 100)} % of CoT ## "
    else:
        prefix_str = f" ## removed tokens: {(removed_indices - cot_start).tolist()} ## " # for sanity check

    prefix = tokenizer(
        prefix_str,
        add_special_tokens=True,
        truncation=True,
        return_tensors="pt"
    )["input_ids"].to(device).squeeze(0)

    eos = torch.full((1,), tokenizer.eos_token_id, dtype=torch.long).to(device)
    prefix = torch.cat([prefix, eos])

    ignored_prefix_labels = torch.full_like(prefix, -100)
    return prefix, ignored_prefix_labels


def apply_cot_masking(input_ids, tokenizer, max_cot_length, device, masking_type="contiguous"):
    """
    Apply CoT masking based on max_cot_length.
    
    Args:
        input_ids: Full sequence with question <eos> cot <eos> answer <eos>
        tokenizer: Tokenizer
        max_cot_length: Maximum CoT length allowed (M)
        device: Device
        masking_type: "left-to-right" or "contiguous" or "random"
    
    Returns:
        modified_input_ids: Input with prefix inserted before first EOS
        prefix: The prefix that was inserted
    """
    batch_size = input_ids.shape[0]
    first_sep_positions, second_sep_positions, eos_positions = _get_sep_positions(input_ids, tokenizer)
    
    modified_inputs = []
    prefixes = []
    
    for batch_idx in range(batch_size):
        # Calculate CoT length: N = distance between second EOS and first EOS
        cot_length = second_sep_positions[batch_idx] - first_sep_positions[batch_idx] - 1
        N = cot_length.item()
        
        # Calculate how many tokens to remove: N - M
        n_tokens_to_remove = max(0, N - max_cot_length)
        removal_p = n_tokens_to_remove / N if N > 0 else 0
        
        if n_tokens_to_remove == 0:
            prefix, _ = _get_prefix_contiguous(tokenizer, 0, 0, device)
        else:
            if masking_type == "left-to-right":
                prefix, _ = _get_prefix_contiguous(tokenizer, n_tokens_to_remove, removal_p, device, left_to_right=True)
            elif masking_type == "contiguous":
                prefix, _ = _get_prefix_contiguous(tokenizer, n_tokens_to_remove, removal_p, device, left_to_right=False)
            else:
                prefix, _ = _get_prefix_random_masking(tokenizer, None, None, removal_p, device)
        
        cot_start = first_sep_positions[batch_idx] + 1
        modified_input = torch.cat([
            input_ids[batch_idx, :first_sep_positions[batch_idx]],
            prefix,
            input_ids[batch_idx, cot_start:]
        ])
        
        modified_inputs.append(modified_input)
        prefixes.append(prefix)
    
    # Pad sequences to same length
    max_len = max(seq.shape[0] for seq in modified_inputs)
    padded_inputs = []
    for seq in modified_inputs:
        if seq.shape[0] < max_len:
            padding = torch.full((max_len - seq.shape[0],), tokenizer.eos_token_id, 
                               dtype=torch.long, device=device)
            seq = torch.cat([seq, padding])
        padded_inputs.append(seq)
    
    modified_input_ids = torch.stack(padded_inputs)
    return modified_input_ids, prefixes


@torch.no_grad()
def generate_distillation_data(dataloader, model, tokenizer, device, args):
    """
    Generate CoT and answers with logits for the whole dataset for distillation.
    """
    model.eval()
    
    all_outputs = []
    all_logits = []
    all_original_inputs = []
    
    print(f"Generating distillation data for {len(dataloader)} batches...")
    
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
        input_ids_full = batch['input_ids'].to(device)
        
        # Split by first EOS using get_sep_positions to not pass CoT and answer into model
        first_sep_positions = get_sep_position(input_ids_full, tokenizer.eos_token_id)
        input_ids_for_generation = input_ids_full[:, :first_sep_positions.max()+1]  # Only question + first EOS
        
        # Apply CoT masking/prefix insertion based on max_cot_length
        if args.max_cot_length > 0:
            input_ids_for_generation, prefixes = apply_cot_masking(
                input_ids_full, tokenizer, args.max_cot_length, device, 
                masking_type=args.masking_type
            )
            # Update input_ids to only include up to first EOS + prefix
            first_sep_positions_new = get_sep_position(input_ids_for_generation, tokenizer.eos_token_id)
            input_ids_for_generation = input_ids_for_generation[:, :first_sep_positions_new.max()+1]
        
        # Generate with return_logits=True
        generation_outputs = model.generate(
            input_ids=input_ids_for_generation,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            stop_on_two_eos=True,
            max_cot_tokens=args.max_cot_length,
            return_logits=True
        )

        if isinstance(generation_outputs, tuple):
            generated_sequences, generated_logits = generation_outputs
        else:
            generated_sequences = generation_outputs
            generated_logits = None
        
        # Store results
        all_outputs.append(generated_sequences.cpu())
        if generated_logits is not None:
            all_logits.append(generated_logits.cpu())
        all_original_inputs.append(input_ids_full.cpu())
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    return {
        'generated_sequences': all_outputs,
        'generated_logits': all_logits if all_logits else None,
        'original_inputs': all_original_inputs
    }


def parse_tuple_list(arg):
    try:
        pairs = [pair.strip("()").split(",") for pair in arg.split()]
        if "." in pairs[0][1]:
            return [(int(x), float(y)) for x, y in pairs]
        return [(int(x), int(y)) for x, y in pairs]
    except ValueError:
        raise argparse.ArgumentTypeError("Schedule must be in the format '(a,b)' with integers.")


def main():
    parser = argparse.ArgumentParser(description="Generate CoT distillation data with logits")
    
    # Model arguments (copied from train.py)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--n_head', type=int, default=None)
    
    # Data arguments
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
    parser.add_argument('--json_dataset', action='store_true',
                        help="Whether dataset is stored in .json (list of dicts with question/answer fields), default is row-wise .txt")
    parser.add_argument('--data_files', type=str, default=None)
    parser.add_argument('--max_samples', type=str, default=None)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--shuffle_seed', type=int, default=None)
    parser.add_argument('--question_key', type=str, default="question")
    parser.add_argument('--answer_key', type=str, default="answer")
    
    # Padding arguments
    parser.add_argument('--pad_cot', action='store_true', default=False)
    parser.add_argument('--max_cot_length', type=int, default=128, 
                        help="Maximum number of tokens allowed in CoT (M)")
    parser.add_argument('--cot_pad_id', type=int, default=None)
    parser.add_argument('--pad_query', action='store_true', default=False)
    parser.add_argument('--max_query_length', type=int, default=128)
    parser.add_argument('--query_pad_id', type=int, default=None)
    
    # Generation arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--masking_type', type=str, choices=['contiguous', 'random'], default='contiguous',
                        help="Type of masking to apply when CoT length exceeds max_cot_length")
    
    # Model loading arguments
    parser.add_argument('--from_pretrained', type=str, default=None, help="Load model weights only")
    parser.add_argument('--from_pretrained_checkpoint', type=str, default=None, help="Load model weights and training state from checkpoint")
    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--use_peft', action='store_true', default=False)
    parser.add_argument('--flash_attention_2', action='store_true', default=False)
    parser.add_argument('--keep_position', action='store_true', default=False)
    parser.add_argument('--reinitialize_weights', action='store_true', default=False)
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Directory to save generated distillation data")
    parser.add_argument('--output_name', type=str, default="distillation_data",
                        help="Name prefix for output files")
    
    # Data processing arguments
    parser.add_argument('--removal_type', type=str, choices=['step-by-step', 'random-chunks', 'random-masks'], default='random-masks')
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--max_size', type=int, default=-1)
    parser.add_argument('--train_max_size', type=int, default=None)
    parser.add_argument('--val_max_size', type=int, default=None)
    parser.add_argument('--test_max_size', type=int, default=None)
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--num_new_tokens', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.from_pretrained is None and args.from_pretrained_checkpoint is None:
        parser.error("Must specify either --from_pretrained or --from_pretrained_checkpoint")
    
    if args.from_pretrained is not None and args.from_pretrained_checkpoint is not None:
        parser.error("Cannot specify both --from_pretrained and --from_pretrained_checkpoint")
    
    print(args)
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model - handle both --from_pretrained and --from_pretrained_checkpoint
    if args.from_pretrained_checkpoint is not None:
        config, model, tokenizer = load_model_from_checkpoint(args.from_pretrained_checkpoint, device)
    elif args.from_pretrained is not None:
        config, model, tokenizer = create_model(args, device)
    else:
        raise ValueError("Must specify either --from_pretrained or --from_pretrained_checkpoint")
    
    if args.removal_type == "random-chunks":
        model, tokenizer, new_token_ids = add_new_tokens(model, tokenizer, args.num_new_tokens + 1)
        start_id, new_token_ids = new_token_ids[0], new_token_ids[1:]
    else:
        new_token_ids = None
    
    print("MODEL ARCHITECTURE:")
    print(model)
    
    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(args, tokenizer, new_token_ids)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate distillation data for each split
    for split_name, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        if dataloader is None:
            continue
            
        print(f"\n=== Generating {split_name} split ===")
        
        results = generate_distillation_data(dataloader, model, tokenizer, device, args)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{args.output_name}_{split_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved {split_name} results to {output_file}")
        print(f"Generated sequences: {len(results['generated_sequences'])} batches")
        if results['generated_logits'] is not None:
            print(f"Generated logits: {len(results['generated_logits'])} batches")
    
    # Generate test split if available
    if test_dataloader is not None:
        print(f"\n=== Generating test split ===")
        
        results = generate_distillation_data(test_dataloader, model, tokenizer, device, args)
        
        output_file = os.path.join(args.output_dir, f"{args.output_name}_test.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved test results to {output_file}")

    print(f"\nAll distillation data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
