#!/usr/bin/env python3
"""
Test script for SubProsQATokenizer with graph dataset.
Tests tokenization of input_ids and labels, including handling of -100 labels.
"""

import sys
import os
import json
import torch
from pathlib import Path

from src.subprosqa.patch_vocabulary import SubProsQATokenizer
from src.dataset.data_chunked import CoTDatasetChunks

def test_tokenizer_basic():
    """Test basic tokenizer functionality."""
    print("=== Testing Basic Tokenizer Functionality ===")
    
    # Create tokenizer with 10000 concepts
    tokenizer = SubProsQATokenizer(num_concepts=10000)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # Test basic tokenization
    test_text = "concept0001->concept0002, concept0003->concept0004"
    print(f"\nTest text: {test_text}")
    
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    input_ids = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Input IDs: {input_ids}")
    
    # Test decoding
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Decoded (with special tokens): {decoded}")
    
    decoded_clean = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"Decoded (clean): {decoded_clean}")
    
    return tokenizer

def test_tokenizer_with_labels():
    """Test tokenizer with labels containing -100 values."""
    print("\n=== Testing Tokenizer with Labels containing -100 ===")
    
    tokenizer = SubProsQATokenizer(num_concepts=10000)
    
    # Create sample input_ids and labels with -100
    text = "concept0001->concept0002, concept0003->concept0004"
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    
    # Create labels with some -100 values (simulating masked tokens)
    labels = input_ids.copy()
    labels[2:4] = -100  # Mask some tokens
    
    print(f"Input IDs: {input_ids}")
    print(f"Labels: {labels}")
    
    # Test decoding input_ids
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"Decoded input: {decoded_input}")
    
    # Test decoding labels (should handle -100 properly)
    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
    print(f"Decoded labels: {decoded_labels}")
    
    # Test decoding with -100 handling
    valid_labels = labels[labels != -100]
    decoded_valid_labels = tokenizer.decode(valid_labels, skip_special_tokens=True)
    print(f"Decoded valid labels only: {decoded_valid_labels}")

def test_dataset_loading():
    """Test loading the graph dataset with CoTDatasetChunks."""
    print("\n=== Testing Dataset Loading ===")
    
    # Dataset path
    dataset_path = "/scratch/ss19021/Internalize_CoT_Step_by_Step/data/shared_graph_cot_3_6_10000n_50000e_100000s_25c_092425/train_subprosqa_dataset.json"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Please check the path or provide the correct path to the dataset.")
        return None
    
    # Create tokenizer
    tokenizer = SubProsQATokenizer(num_concepts=10000)
    
    # Load dataset with default parameters
    try:
        dataset = CoTDatasetChunks(
            tokenizer=tokenizer,
            path=dataset_path,
            json_dataset=True,
            max_length=-1,
            max_size=5,  # Load only 5 samples for testing
            chunk_size=8,
            num_new_tokens=1000,
            new_token_ids=list(range(10000, 11000)),  # Mock new token IDs
            pad_cot=False,
            max_cot_length=-1,
            cot_pad_id=None,
            pad_query=False,
            max_query_length=-1,
            query_pad_id=None,
            dont_mask_question_in_labels=False
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        return dataset, tokenizer
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def test_sample_tokenization(dataset, tokenizer):
    """Test tokenization on actual dataset samples."""
    print("\n=== Testing Sample Tokenization ===")
    
    if dataset is None:
        print("No dataset available for testing")
        return
    
    # Test first few samples
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Input IDs: {input_ids.tolist()}")
        print(f"Labels: {labels.tolist()}")
        
        # Decode input_ids
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Decoded input: {decoded_input}")
        
        # Decode labels (handling -100)
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0:
            decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=True)
            print(f"Decoded valid labels: {decoded_labels}")
        else:
            print("All labels are -100 (masked)")
        
        # Show chunk information if available
        if sample["chunk_positions"] is not None:
            print(f"Chunk positions: {sample['chunk_positions']}")
        if sample["chunk_input_ids"] is not None:
            print(f"Chunk input IDs: {sample['chunk_input_ids'].tolist()}")

def main():
    """Main test function."""
    print("Testing SubProsQATokenizer with Graph Dataset")
    print("=" * 50)
    
    # Test basic tokenizer functionality
    tokenizer = test_tokenizer_basic()
    
    # Test tokenizer with -100 labels
    test_tokenizer_with_labels()
    
    # Test dataset loading
    dataset, tokenizer = test_dataset_loading()
    
    # Test sample tokenization
    test_sample_tokenization(dataset, tokenizer)
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    main()
