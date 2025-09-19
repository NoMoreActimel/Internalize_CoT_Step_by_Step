import gc
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

        self.chunks = []
        self.samples = []
        self._read_chunk_files()

        self.next_chunk_index = 0
        self.current_index_within_chunk = 0
        self.current_chunk_length = 0

        self._load_next_chunk()
    
    def _read_chunk_files(self):
        print(f"Loading distillation data from all chunk pickle files in {self.data_dir}")
        chunk_pattern = os.path.join(self.data_dir, f"chunk_*.pkl")
        self.chunk_files = sorted(glob.glob(chunk_pattern))
        print(f"Found {len(self.chunk_files)} chunk files!")

    def _load_next_chunk(self):
        del self.samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.samples = []

        chunk_file = self.chunk_files[self.next_chunk_index]
        print(f"Loading chunk with index {self.next_chunk_index + 1} / {len(self.chunk_files)}: {chunk_file}")
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
        
        self.next_chunk_index += 1
        self.current_index_within_chunk = 0
        self.current_chunk_length = len(chunk_data['generated_sequences'])

        for i in range(self.current_chunk_length):
            self.samples.append({
                'generated_sequence': chunk_data['generated_sequences'][i],
                'teacher_logits': chunk_data['generated_logits'][i],
                'original_input': chunk_data['original_inputs'][i]
            })
        print(f"Loaded {len(self.samples)} samples for distillation from {chunk_file}!")

    def __len__(self):
        return len(self.samples) * len(self.chunk_files)
    
    def __getitem__(self, idx):
        print("[Distillation dataset] index received:", idx)
        print("[Distillation dataset] current chunk index:", self.next_chunk_index - 1)
        print("[Distillation dataset] current index within chunk:", self.current_index_within_chunk)
 
        # sample = self.samples[idx]
        sample = self.samples[self.current_index_within_chunk]
        self.current_index_within_chunk += 1
        if self.current_index_within_chunk >= self.current_chunk_length:
            self._load_next_chunk()
        
        # Get teacher's generated sequence (input_ids)
        generated_sequence = sample['generated_sequence']
        
        # Truncate if needed
        if self.max_length and generated_sequence.shape[0] > self.max_length:
            generated_sequence = generated_sequence[:self.max_length]

        # Cut <cot> from right
        cot_id = self.tokenizer.encode("<cot>", add_special_tokens=False)[0]
        last = (generated_sequence != cot_id).nonzero(as_tuple=True)[0][-1].item() + 1
        generated_sequence = generated_sequence[:last]
        
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
            teacher_logits = teacher_logits[:last]
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
        max_len = max(f['input_ids'].shape[0] for f in features)
        
        if self.pad_to_multiple_of:
            max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            max_len = max_len * self.pad_to_multiple_of
        
        batch_size = len(features)
        vocab_size = None
        
        has_teacher_logits = 'teacher_logits' in features[0]
        if has_teacher_logits:
            vocab_size = features[0]['teacher_logits'].shape[-1]
        
        input_ids = torch.full((batch_size, max_len), self.tokenizer.eos_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        if has_teacher_logits:
            teacher_logits = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.float)
        
        # Fill in the data
        for i, feature in enumerate(features):
            seq_len = feature['input_ids'].shape[0]
            
            input_ids[i, :seq_len] = feature['input_ids']
            labels[i, :seq_len] = feature['labels']
            
            print(f"input_ids.shape: {input_ids.shape}")
            print(f"labels.shape: {labels.shape}")
            print(f"seq_len: {seq_len}")

            if has_teacher_logits:
                print(f"logits.shape: {feature['teacher_logits'].shape}")
                # input_ids are batched already, teacher logits are not
                if feature['teacher_logits'].shape[0] < seq_len:
                    pad_shape = list(feature['teacher_logits'].shape)
                    pad_shape[0] = seq_len - feature['teacher_logits'].shape[0]
                    pad_tensor = torch.full(
                        pad_shape, -100.0,
                        dtype=feature['teacher_logits'].dtype,
                        device=feature['teacher_logits'].device
                    )
                    feature['teacher_logits'] = torch.cat([pad_tensor, feature['teacher_logits']], dim=0)
                    print(f"Padded teacher logits, final shape: {feature['teacher_logits'].shape}")
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

