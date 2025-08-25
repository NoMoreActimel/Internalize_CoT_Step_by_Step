import glob
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class DistillationDataset(Dataset):
    """
    Dataset for loading teacher generations and logits for distillation.
    
    The dataset expects pickle files created by generate_distillation.py containing:
    - generated_sequences: Teacher model output sequences
    - generated_logits: Teacher model logits over vocabulary  
    - original_inputs: Original input sequences
    """
    
    def __init__(self, data_dir, tokenizer, max_length=None):
        """
        Args:
            pickle_path: Path to pickle file with teacher generations
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length (optional truncation)
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self._load_all_chunks()
    
    def _load_all_chunks(self):
        print(f"Loading distillation data from all chunk pickle files in {self.data_dir}")
        chunk_pattern = os.path.join(self.data_dir, f"chunk_*.pkl")
        chunk_files = sorted(glob.glob(chunk_pattern))
        print(f"Found {len(self.chunk_files)} chunk files!")
        for chunk_file in chunk_files:
            print(f"Loading chunk: {chunk_file}")
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            for i in range(len(chunk_data['generated_sequences'])):
                self.samples.append({
                    'generated_sequence': chunk_data['generated_sequences'][i],
                    'teacher_logits': chunk_data['generated_logits'][i],
                    'original_input': chunk_data['original_inputs'][i]
                })
        print(f"Loaded {len(self.samples)} samples for distillation from {self.data_dir}!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get teacher's generated sequence (input_ids)
        generated_sequence = sample['generated_sequence']
        
        # Truncate if needed
        if self.max_length and generated_sequence.shape[0] > self.max_length:
            generated_sequence = generated_sequence[:self.max_length]
        
        # Create labels (shifted by 1 for next token prediction)
        labels = generated_sequence.clone()
        
        result = {
            'input_ids': generated_sequence,
            'labels': labels,
        }
        
        # Add teacher logits if available
        if 'teacher_logits' in sample:
            teacher_logits = sample['teacher_logits']
            if self.max_length and teacher_logits.shape[0] > self.max_length:
                teacher_logits = teacher_logits[:self.max_length]
            result['teacher_logits'] = teacher_logits
        
        return result


class DistillationDataCollator:
    """
    Data collator for distillation dataset that handles padding.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Get maximum length in the batch
        max_len = max(f['input_ids'].shape[0] for f in features)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch_size = len(features)
        vocab_size = None
        
        # Check if we have teacher logits
        has_teacher_logits = 'teacher_logits' in features[0]
        if has_teacher_logits:
            vocab_size = features[0]['teacher_logits'].shape[-1]
        
        # Initialize batch tensors
        input_ids = torch.full((batch_size, max_len), self.tokenizer.eos_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        if has_teacher_logits:
            teacher_logits = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.float)
        
        # Fill in the data
        for i, feature in enumerate(features):
            seq_len = feature['input_ids'].shape[0]
            
            input_ids[i, :seq_len] = feature['input_ids']
            labels[i, :seq_len] = feature['labels']
            
            if has_teacher_logits:
                teacher_logits[i, :seq_len] = feature['teacher_logits']
        
        result = {
            'input_ids': input_ids,
            'labels': labels,
        }
        
        if has_teacher_logits:
            result['teacher_logits'] = teacher_logits
        
        return result


def load_distillation_data(train_dir_path, val_dir_path, tokenizer, batch_size, 
                          max_length=None, test_dir_path=None):
    """
    Load distillation datasets and create dataloaders.
    
    Args:
        train_dir_path: Path to training pickle file
        val_dir_path: Path to validation pickle file  
        tokenizer: Tokenizer to use
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        test_dir_path: Optional path to test pickle file
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    
    # Create datasets
    train_dataset = DistillationDataset(train_dir_path, tokenizer, max_length)
    val_dataset = DistillationDataset(val_dir_path, tokenizer, max_length)
    
    # Create collator
    collate_fn = DistillationDataCollator(tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=False
    )
    
    test_dataloader = None
    if test_dir_path and os.path.exists(test_dir_path):
        test_dataset = DistillationDataset(test_dir_path, tokenizer, max_length)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn, 
            shuffle=False
        )
    
    return train_dataloader, val_dataloader, test_dataloader
