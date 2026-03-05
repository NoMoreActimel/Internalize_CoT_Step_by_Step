import math
import argparse
import os
import gc
import sys
import tqdm
import inspect
import logging
import random
import numpy as np
import torch
import pickle

from src.dataset.data import load_data
from src.dataset.data_chunked import add_new_tokens
from src.model import ImplicitModel
from src.model_jepa import JEPAImplicitModel
from src.configuration_model import ImplicitModelConfig
from src.utils import get_sep_position, extract_answer, extract_cot
from src.utils import expand_gpt2_positions
from src.create_model import create_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def load_teacher_model_from_checkpoint(checkpoint_path, device):
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


def analyze_generation_stats(sequences, tokenizer):
    """
    Analyze CoT and answer lengths from generated sequences.
    
    Args:
        sequences: List of generated sequence tensors
        tokenizer: Tokenizer
    
    Returns:
        dict: Statistics about CoT and answer lengths
    """
    cot_lengths = []
    answer_lengths = []
    total_lengths = []
    
    for seq in sequences:
        # Find EOS positions - structure is: question <eos> cot <eos> answer <eos>
        eos_positions = (seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        
        if len(eos_positions) >= 2:
            # First EOS ends question, second EOS ends CoT
            first_eos = eos_positions[0].item()
            second_eos = eos_positions[1].item()
            
            # CoT length is between first and second EOS
            cot_length = second_eos - first_eos - 1
            cot_lengths.append(cot_length)
            
            # Answer length is from second EOS to end (or third EOS if exists)
            if len(eos_positions) >= 3:
                third_eos = eos_positions[2].item()
                answer_length = third_eos - second_eos - 1
            else:
                # No third EOS, answer goes to end of sequence
                answer_length = len(seq) - second_eos - 1
            
            answer_lengths.append(answer_length)
            total_lengths.append(len(seq))
        else:
            # Incomplete generation - just track total length
            total_lengths.append(len(seq))
    
    stats = {
        'cot_lengths': {
            'mean': np.mean(cot_lengths) if cot_lengths else 0,
            'std': np.std(cot_lengths) if cot_lengths else 0,
            'min': min(cot_lengths) if cot_lengths else 0,
            'max': max(cot_lengths) if cot_lengths else 0,
            'count': len(cot_lengths)
        },
        'answer_lengths': {
            'mean': np.mean(answer_lengths) if answer_lengths else 0,
            'std': np.std(answer_lengths) if answer_lengths else 0,
            'min': min(answer_lengths) if answer_lengths else 0,
            'max': max(answer_lengths) if answer_lengths else 0,
            'count': len(answer_lengths)
        },
        'total_lengths': {
            'mean': np.mean(total_lengths) if total_lengths else 0,
            'std': np.std(total_lengths) if total_lengths else 0,
            'min': min(total_lengths) if total_lengths else 0,
            'max': max(total_lengths) if total_lengths else 0,
            'count': len(total_lengths)
        },
        'incomplete_generations': len(sequences) - len(cot_lengths)
    }
    
    return stats

def print_generation_stats(stats, split_name):
    """Print generation statistics in a readable format."""
    print(f"\n=== {split_name.upper()} Generation Statistics ===")
    print(f"Total samples: {stats['total_lengths']['count']}")
    print(f"Complete generations: {stats['cot_lengths']['count']}")
    print(f"Incomplete generations: {stats['incomplete_generations']}")
    
    if stats['cot_lengths']['count'] > 0:
        print(f"\nCoT Lengths:")
        print(f"  Mean: {stats['cot_lengths']['mean']:.1f} ± {stats['cot_lengths']['std']:.1f}")
        print(f"  Range: {stats['cot_lengths']['min']} - {stats['cot_lengths']['max']}")
        
        print(f"\nAnswer Lengths:")
        print(f"  Mean: {stats['answer_lengths']['mean']:.1f} ± {stats['answer_lengths']['std']:.1f}")
        print(f"  Range: {stats['answer_lengths']['min']} - {stats['answer_lengths']['max']}")
    
    print(f"\nTotal Sequence Lengths:")
    print(f"  Mean: {stats['total_lengths']['mean']:.1f} ± {stats['total_lengths']['std']:.1f}")
    print(f"  Range: {stats['total_lengths']['min']} - {stats['total_lengths']['max']}")




class DistillationGenerator:
    def __init__(self, args):
        self.args = args
        self.args.disable_train_shuffle = True

        self.chunk_size = getattr(self.args, "chunk_size", 1024)
        self.batch_size = getattr(self.args, "batch_size", 32)
        assert self.chunk_size % self.batch_size == 0, "chunk_size must be divisible by batch_size"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.from_pretrained_checkpoint is not None:
            _, self.model, self.tokenizer = load_teacher_model_from_checkpoint(self.args.from_pretrained_checkpoint, self.device)
        elif self.args.from_pretrained is not None:
            _, self.model, self.tokenizer = create_model(self.args, self.device)
        else:
            raise ValueError("Must specify either --from_pretrained or --from_pretrained_checkpoint")

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.create_dataloaders()
        self.dataloaders = {"train": self.train_dataloader, "val": self.val_dataloader, "test": self.test_dataloader}

    def create_dataloaders(self):    
        if getattr(self.args, "removal_type", None) == "random-chunks":
            self.model, self.tokenizer, new_token_ids = add_new_tokens(self.model, self.tokenizer, self.args.num_new_tokens + 1)
            start_id, new_token_ids = new_token_ids[0], new_token_ids[1:]
        else:
            new_token_ids = None
        
        self.args.train_type = 'full-cot' # for correct original dataset class - CoTDatasetChunks
        return load_data(self.args, self.tokenizer, new_token_ids)

    def apply_cot_masking(self, input_ids, tokenizer, max_cot_length, device, masking_type="contiguous", return_without_cot=False):
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
            if return_without_cot:
                modified_input = torch.cat([
                    input_ids[batch_idx, :first_sep_positions[batch_idx]],
                    prefix
                ])
            else:
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

    def generate_distillation_data(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        chunk_files = {}
        for split in ["train", "val", "test"]:
            chunk_files[split] = self.generate_split_distillation_data(split)
        return chunk_files

    @torch.no_grad()
    def generate_split_distillation_data(self, split_name, num_batches=None, first_batch_idx=0):
        """
        Generate CoT and answers with logits for the whole dataset for distillation.
        """
        self.model.eval()
        
        chunk_outputs = []
        chunk_logits = []
        chunk_original_inputs = []

        dataloader = self.dataloaders[split_name]
        num_batches = len(dataloader) if num_batches is None else num_batches

        sample_count = first_batch_idx * self.batch_size
        chunk_idx = sample_count // self.chunk_size  # init checks chunk_size is divisible by batch_size
        chunk_files = []

        split_output_dir = f"{self.args.output_dir}/{split_name}"
        os.makedirs(split_output_dir, exist_ok=True)
        
        print(f"Generating distillation data for split {split_name} for {num_batches} batches starting from batch {first_batch_idx}...")
        
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
            if batch_idx < first_batch_idx:
                continue
            if batch_idx >= first_batch_idx + num_batches:
                break

            input_ids_full = batch['input_ids'].to(self.device)
            
            first_sep_positions = get_sep_position(input_ids_full, self.tokenizer.eos_token_id)
            input_ids_for_generation = input_ids_full[:, :first_sep_positions.max()+1]
            
            if self.args.max_cot_length > 0:
                input_ids_masked, prefixes = self.apply_cot_masking(
                    input_ids_full, self.tokenizer, self.args.max_cot_length, self.device, 
                    masking_type=self.args.masking_type,
                    return_without_cot=False
                )
                input_ids_for_generation, prefixes = self.apply_cot_masking(
                    input_ids_full, self.tokenizer, self.args.max_cot_length, self.device, 
                    masking_type=self.args.masking_type,
                    return_without_cot=True
                )
                # Update input_ids to only include up to first EOS + prefix
                # first_sep_positions_new = get_sep_position(input_ids_masked, self.tokenizer.eos_token_id)
                # input_ids_for_generation = input_ids_masked[:, :first_sep_positions_new.max()+1]
            
            generation_outputs = self.model.generate(
                input_ids=input_ids_for_generation,
                max_new_tokens=self.args.max_new_tokens,
                num_beams=1,
                stop_on_two_eos=True,
                max_cot_tokens=self.args.max_cot_length,
                inputs_with_cot=False,
                return_logits=True
            )

            if isinstance(generation_outputs, tuple):
                generated_sequences, generated_logits = generation_outputs
            else:
                generated_sequences = generation_outputs
                generated_logits = None
            
            sequence_samples, logit_samples = self.flatten_batch_to_samples(generated_sequences, generated_logits)

            print("Generation examples:")
            for i in range(input_ids_full.shape[0]):
                chunk_original_inputs.append(input_ids_full[i].cpu())
                chunk_outputs.append(sequence_samples[i].cpu())
                chunk_logits.append(logit_samples[i].cpu())
                sample_count += 1

                if i < 3:
                    gen_inputs = self.tokenizer.decode(input_ids_for_generation[i], skip_special_tokens=False)
                    orig_text = self.tokenizer.decode(input_ids_full[i], skip_special_tokens=False)
                    gen_text = self.tokenizer.decode(sequence_samples[i], skip_special_tokens=False)
                    print(f"Input: {gen_inputs}")
                    print(f"Target: {orig_text}")
                    if self.args.max_cot_length > 0:
                        masked_text = self.tokenizer.decode(input_ids_masked[i], skip_special_tokens=False)
                        print(f"Masked: {masked_text}")
                    print(f"Predicted: {gen_text}")
                print("-" * 50)

            if sample_count >= self.chunk_size:
                chunk_stats = analyze_generation_stats(chunk_outputs, self.tokenizer)
                print_generation_stats(chunk_stats, split_name)
                
                chunk_data = {
                    'generated_sequences': chunk_outputs,
                    'generated_logits': chunk_logits if chunk_logits else None,
                    'original_inputs': chunk_original_inputs,
                    'generation_stats': chunk_stats,
                    'chunk_idx': chunk_idx,
                    'sample_count': len(chunk_outputs)
                }
                
                chunk_files.append(self.save_chunk(chunk_data, split_output_dir, chunk_idx))
                
                print(f"Chunk {chunk_idx}: {len(chunk_outputs)} samples, "
                    f"avg CoT length: {chunk_stats['cot_lengths']['mean']:.1f}")
                
                chunk_outputs = []; chunk_logits = []; chunk_original_inputs = []; sample_count = 0
                chunk_idx += 1
                
                gc.collect()
                torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}, samples: {sample_count + chunk_idx * self.chunk_size}")
        
        if chunk_outputs:
            chunk_stats = analyze_generation_stats(chunk_outputs, self.tokenizer)
            print_generation_stats(chunk_stats, split_name)
            
            chunk_data = {
                'generated_sequences': chunk_outputs,
                'generated_logits': chunk_logits if chunk_logits else None,
                'original_inputs': chunk_original_inputs,
                'generation_stats': chunk_stats,
                'chunk_idx': chunk_idx,
                'sample_count': len(chunk_outputs)
            }
            
            chunk_file = self.save_chunk(chunk_data, split_output_dir, chunk_idx)
            chunk_files.append(chunk_file)
            
            print(f"Final chunk {chunk_idx}: {len(chunk_outputs)} samples")
        
        return chunk_files

    def flatten_batch_to_samples(self, sequences, logits=None):
        """
        sequences: Either a tensor [batch_size, seq_len] or list of tensors
        logits: Either a tensor [batch_size, seq_len, vocab_size] or list of tensors, or None
        return: (list of sequence tensors, list of logit tensors or None)
        """
        sequence_samples = []
        logit_samples = []
        
        if isinstance(sequences, list):
            sequence_samples = sequences
            if logits is not None:
                for i in range(len(sequences)):
                    logit_samples.append(logits[i])
        else:
            for i in range(sequences.shape[0]):
                sequence_samples.append(sequences[i])
                if logits is not None:
                    logit_samples.append(logits[i])
        
        return sequence_samples, logit_samples if logits is not None else None


    def save_chunk(self, chunk_data, output_dir, chunk_idx):
        """Save a chunk of data to disk."""
        chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk_data, f)
        print(f"Saved chunk {chunk_idx} to {chunk_file}")
        return chunk_file
