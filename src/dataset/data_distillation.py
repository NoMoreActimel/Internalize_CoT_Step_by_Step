import gc
import glob
import json
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.distillation_generator import DistillationGenerator

class Args:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'Args({attrs})'

class DistillationDataset(Dataset):
    """
    Dataset for loading teacher generations and logits for distillation.
    
    The dataset expects pickle files created by generate_distillation.py containing:
    - generated_sequences: Teacher model output sequences
    - generated_logits: Teacher model logits over vocabulary  
    - original_inputs: Original input sequences

    Alternatively, the dataset calls the generator used in generate_distillation.py in process,
    storing at most chunk_storage_limit chunks in memory.
    """
    
    def __init__(
        self,
        data_dir,
        tokenizer,
        max_length=None,
        generate_in_process=False,
        generator_args_path=None,
        chunk_storage_limit=None,
        split=None
    ):
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

        self.generate_in_process = generate_in_process
        self.generator_args_path = generator_args_path
        self.chunk_storage_limit = chunk_storage_limit

        if split is None:
            for name in ["train", "val", "test"]:
                if name in self.data_dir.split("/")[-1]:
                    split = name
                    break
            if split is None:
                raise ValueError(f"Split not provided and not found in {self.data_dir}")
        self.split = split

        if self.generate_in_process:
            generator_config_dict = json.load(open(self.generator_args_path, 'r'))
            self.generator_args = Args(generator_config_dict)
            self.generator = DistillationGenerator(self.generator_args)
            self.batch_size = self.generator_args.batch_size
            self.current_batch_index = 0
            self.total_batches = len(self.generator.dataloaders[self.split])
        else:
            self._read_chunk_files()

        self.next_chunk_index = -1
        self.current_index_within_chunk = 0
        self.current_chunk_length = 0

        self._load_next_chunk()
    
    def _read_chunk_files(self):
        print(f"Loading distillation data from all chunk pickle files in {self.data_dir}")
        chunk_pattern = os.path.join(self.data_dir, f"chunk_*.pkl")
        self.chunk_files = sorted(glob.glob(chunk_pattern))
        print(f"Found {len(self.chunk_files)} chunk files!")

    def _prepare_next_chunks(self):
        if self.generate_in_process:
            self._generate_next_chunks()
        return self._load_next_chunk()
    
    def _generate_next_chunks(self):
        # Delete old chunk files if they exist
        for chunk_file in self.chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            
        self.chunk_files = self.generator.generate_split_distillation_data(
            split_name=self.split,
            num_batches=self.chunk_storage_limit // self.batch_size,
            first_batch_idx=self.current_batch_index
        )

        self.current_batch_index += self.chunk_storage_limit // self.batch_size
        if self.current_batch_index >= self.total_batches:
            self.current_batch_index = 0

    def _load_next_chunk(self):
        del self.samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.samples = []

        self.next_chunk_index += 1
        self.current_index_within_chunk = 0
        if self.next_chunk_index >= len(self.chunk_files):
            self.next_chunk_index = 0
            if self.generate_in_process:
                self._generate_next_chunks()

        chunk_file = self.chunk_files[self.next_chunk_index]
        print(f"Loading chunk with index {self.next_chunk_index + 1} / {len(self.chunk_files)}: {chunk_file}")
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
        
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


def load_distillation_data(args, tokenizer):
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
        #     train_dir_path=args.train_path,
        #     val_dir_path=args.val_path,
        #     tokenizer=tokenizer,
        #     batch_size=args.batch_size,
        #     max_length=args.max_len_train if args.max_len_train > 0 else None,
        #     test_dir_path=args.test_path
        # )
    
    train_dir_path = getattr(args, 'train_path', None)
    val_dir_path = getattr(args, 'val_path', None)
    test_dir_path = getattr(args, 'test_path', None)
    max_length = getattr(args, 'max_len_train', None) if getattr(args, 'max_len_train', None) > 0 else None
    batch_size = getattr(args, 'batch_size', None)

    generate_in_process = getattr(args, 'distillation_generate_in_process', False)
    generator_args_path = getattr(args, 'distillation_generator_args_path', None)
    chunk_storage_limit = getattr(args, 'distillation_chunk_storage_limit', None)
    
    # Create datasets
    train_dataset = DistillationDataset(
        train_dir_path, tokenizer, max_length,
        generate_in_process, generator_args_path,
        chunk_storage_limit, split="train"
    )
    val_dataset = DistillationDataset(
        val_dir_path, tokenizer, max_length,
        generate_in_process, generator_args_path,
        chunk_storage_limit, split="val"
    )
    
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
        test_dataset = DistillationDataset(
            test_dir_path, tokenizer, max_length,
            generate_in_process, generator_args_path,
            chunk_storage_limit, split="test"
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn, 
            shuffle=False
        )
    
    return train_dataloader, val_dataloader, test_dataloader
