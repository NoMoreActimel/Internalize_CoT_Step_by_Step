import argparse
import random
import torch
from src.dataset.distillation_generator import DistillationGenerator

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
    parser.add_argument('--masking_type', type=str, choices=['left-to-right', 'contiguous', 'random'], default='contiguous',
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
    
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--num_new_tokens', type=int, default=1000)
    
    args = parser.parse_args()
    
    print(args)
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    generator = DistillationGenerator(args)
    generator.generate_distillation_data()
    print(f"\nAll distillation data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
