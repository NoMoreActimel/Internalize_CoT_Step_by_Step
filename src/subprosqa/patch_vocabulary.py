import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel

from src.model import ImplicitModel


# TO-DO:
# - make tokenizer without inheritance. Methods:
#     - encode
#     - batch_encode
#     - batch_decode
#     - __call__
#     - get_vocab
#     - save_pretrained
#     - special tokens + ids
# - change model init, pass tokenizer + model patcher into init

class SubProsQATokenizer:
    """Custom tokenizer for SubProsQA dataset with graph concepts."""
    
    def __init__(self, num_concepts: int, max_len: int = None):
        self._token_ids = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<sep>': 4,
            '<mask>': 5,
            '<cot>': 6,      # CoT pad token
            '<query>': 7,    # Query pad token
            '[EDGES]': 8,    # Edge list marker
            '[FACTS]': 9,    # Facts marker (for hybrid mode)
            'Question:': 10,  # Question marker
            'Answer:': 11,    # Answer marker
            '####': 12,       # Separator in answer format
            '->': 13,         # Edge direction token
            ',': 14,          # Comma separator
            'is': 15,         # "is" relation
            'or': 16,         # "or" in questions
            '?': 17,          # Question mark
            ' ': 18,          # Space token
            '\n': 19,         # Newline token
            '##': 20,         # Double hash marker
            'masked': 21,     # Masked token indicator
            '0': 22,          # Digit 0
            '1': 23,          # Digit 1
            '2': 24,          # Digit 2
            '3': 25,          # Digit 3
            '4': 26,          # Digit 4
            '5': 27,          # Digit 5
            '6': 28,          # Digit 6
            '7': 29,          # Digit 7
            '8': 30,          # Digit 8
            '9': 31,          # Digit 9
            '% of CoT': 32,   # Percentage of CoT indicator
            'from': 33,       # From keyword
        }
        self.num_concepts = num_concepts
        num_special_tokens = len(self._token_ids)
        for i in range(self.num_concepts):
            self._token_ids[f"concept{i:04d}"] = i + num_special_tokens
        
        self._id_tokens = {value: key for key, value in self._token_ids.items()}        
        self._sorted_tokens = sorted(self._token_ids.keys(), key=len, reverse=True)
        self.max_length = len(self._sorted_tokens[0])
        self.min_length = len(self._sorted_tokens[-1])

        self.special_tokens_map = {
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'sep_token': '<sep>',
            'mask_token': '<mask>',
            'cot_pad_token': '<cot>',
            'query_pad_token': '<query>'
        }
        # Special token IDs
        self.pad_token_id = self._token_ids['<pad>']
        self.unk_token_id = self._token_ids['<unk>']
        self.bos_token_id = self._token_ids['<bos>']
        self.eos_token_id = self._token_ids['<eos>']
        self.sep_token_id = self._token_ids['<sep>']
        self.mask_token_id = self._token_ids['<mask>']
        self.cot_pad_token_id = self._token_ids['<cot>']
        self.query_pad_token_id = self._token_ids['<query>']

        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.sep_token = '<sep>'
        self.mask_token = '<mask>'
        self.cot_pad_token = '<cot>'
        self.query_pad_token = '<query>'
    
    def encode(self, text, add_special_tokens=True, truncation=True, max_length=512, padding=False, return_tensors=None, **kwargs):
        """
        Encode text(s) into token IDs. Supports both single strings and lists of strings.
        
        Args:
            text: String or list of strings to encode
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate sequences
            max_length: Maximum length of sequences
            padding: Whether to pad sequences (not implemented yet)
            return_tensors: Type of tensor to return ('pt' for PyTorch, 'np' for numpy, None for lists)
            **kwargs: Additional arguments
        
        Returns:
            List of token IDs or list of lists if multiple texts
        """
        # Handle batched input
        if isinstance(text, list):
            token_ids_list = [
                self.encode(
                    t, add_special_tokens=add_special_tokens, truncation=truncation, 
                    max_length=max_length, padding=padding,
                    return_tensors=return_tensors, **kwargs
                )['input_ids']
                for t in text
            ]
            max_ids_length = max([len(token_ids) for token_ids in token_ids_list])
            if max_length is not None:
                max_ids_length = min(max_length, max_ids_length)
            if return_tensors == 'pt':
                for i, token_ids in enumerate(token_ids_list):
                    pad_ids = torch.full((max_ids_length - len(token_ids),), self.pad_token_id)
                    token_ids_list[i] = torch.cat([token_ids, pad_ids])
                return {"input_ids": torch.stack(token_ids_list)}
            elif return_tensors == 'np':
                for i, token_ids in enumerate(token_ids_list):
                    pad_ids = np.full((max_ids_length - len(token_ids),), self.pad_token_id)
                    token_ids_list[i] = np.concatenate([token_ids, pad_ids])
                return {"input_ids": np.stack(token_ids_list)}
            else:
                for i, token_ids in enumerate(token_ids_list):
                    pad_ids = [self.pad_token_id] * (max_ids_length - len(token_ids))
                    token_ids_list[i] = token_ids + pad_ids
                return {"input_ids": token_ids_list}
        
        # Single text encodingxw
        tokens = []
        position = 0

        if add_special_tokens:
            text = self.bos_token + text + self.eos_token
        
        while position < len(text):
            matched = False
            
            for token_length in range(self.max_length, self.min_length - 1, -1):
                if text[position:position + token_length] in self._token_ids:
                    tokens.append(text[position:position+token_length])
                    position += token_length
                    matched = True
                    break
            
            if not matched:
                if text[position:].startswith('concept'):
                    concept_match = text[position:position+11]  # 'conceptXXXX' is 11 chars
                    raise ValueError(
                        f"Token '{concept_match}' at position {position} not found in vocabulary. Text: '{text}'."
                        f"This tokenizer only supports concepts from concept0000 to concept{self.num_concepts-1:04d}."
                    )
                else:
                    raise ValueError(
                        f"Cannot tokenize text at position {position}. Text: '{text}'.\nProblems at: '{text[position:]}'\n"
                        f"No matching token found in vocabulary."
                    )
        
        # Convert tokens to IDs
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        
        # Handle truncation
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        if return_tensors == 'pt':
            token_ids = torch.tensor(token_ids)
        elif return_tensors == 'np':
            token_ids = np.array(token_ids)
        
        # Return as dictionary with input_ids key to match transformers format
        if return_tensors is not None:
            return {"input_ids": token_ids}
        return token_ids
    
    def batch_encode(self, texts, **kwargs):
        """Alias for encode method to handle batched texts."""
        return self.encode(texts, **kwargs)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_ids.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_tokens.get(index, self.unk_token)
    
    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs):
        """
        Decode token IDs back to text, handling -100 values properly.
        -100 values are ignored during decoding as they represent masked tokens.
        Supports both single sequences and batched sequences.
        """
        # Handle batched input
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            return [
                self.decode(
                    seq, skip_special_tokens=skip_special_tokens, 
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs
                ) 
                for seq in token_ids
            ]
        
        # Handle tensor input
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Filter out -100 values (masked tokens) and convert to tokens
        valid_token_ids = [token_id for token_id in token_ids if token_id != -100]
        
        if not valid_token_ids:
            return ""

        tokens = []
        for token_id in valid_token_ids:
            if token_id in self._id_tokens:
                token = self._id_tokens[token_id]
                if not skip_special_tokens or token not in self.special_tokens_map.values():
                    tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        text = ''.join(tokens)
        
        # Clean up tokenization spaces if requested
        if clean_up_tokenization_spaces:
            # Basic cleanup - remove extra spaces
            text = ' '.join(text.split())
        
        return text
    
    def batch_decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs):
        """Decode multiple sequences of token IDs back to text."""
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens, 
                          clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
    
    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=512, padding=False, return_tensors=None, **kwargs):
        """
        Call method for tokenizer. Alias for encode method.
        """
        return self.encode(text, add_special_tokens=add_special_tokens, truncation=truncation,
                          max_length=max_length, padding=padding, return_tensors=return_tensors, **kwargs)

    def get_vocab(self) -> Dict[str, int]:
        return self._token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory) / (filename_prefix + 'vocab.json')
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vocab_path, 'w') as f:
            json.dump(self._token_ids, f, indent=2)
        
        return (str(vocab_path),)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save tokenizer to a directory in the format expected by transformers library.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_path = save_directory / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(self._token_ids, f, indent=2)
        
        # Save tokenizer config
        config = {
            'vocab_size': self.vocab_size,
            'num_concepts': self.num_concepts,
            'special_tokens_map': self.special_tokens_map,
            'max_length': self.max_length,
            'min_length': self.min_length
        }
        
        config_path = save_directory / 'tokenizer_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save special tokens map
        special_tokens_path = save_directory / 'special_tokens_map.json'
        with open(special_tokens_path, 'w') as f:
            json.dump(self.special_tokens_map, f, indent=2)
        
        return save_directory

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)


def patch_model(
    model: ImplicitModel,
    tokenizer: SubProsQATokenizer,
    initialize_strategy: str = 'random'
) -> ImplicitModel:
    """Patch model to work with custom tokenizer vocabulary."""
    
    old_vocab_size = model.base_model.config.vocab_size
    new_vocab_size = tokenizer.vocab_size
    
    model.change_tokenizer(tokenizer)
    print(f"Resizing model embeddings from {old_vocab_size} to {new_vocab_size}")
    model.base_model.resize_token_embeddings(new_vocab_size)
    # model.base_model.config.vocab_size = new_vocab_size
    model.base_model.config.pad_token_id = tokenizer.pad_token_id
    model.base_model.config.eos_token_id = tokenizer.eos_token_id
    model.base_model.config.bos_token_id = tokenizer.bos_token_id

    # for attr_name in ['lm_head', 'score']:
    #     if hasattr(model.base_model, attr_name) and getattr(model.base_model, attr_name) is not None:
    #         print(f"Resizing {attr_name} layer")       
    #         setattr(model.base_model, attr_name, torch.nn.Linear(
    #             in_features=getattr(model.base_model, attr_name).in_features,
    #             out_features=new_vocab_size,
    #             bias=getattr(model.base_model, attr_name).bias is not None
    #         ))
            
    #         if initialize_strategy == 'random':
    #             torch.nn.init.normal_(getattr(model.base_model, attr_name).weight, mean=0.0, std=0.02)
    #             if getattr(model.base_model, attr_name).bias is not None:
    #                 torch.nn.init.zeros_(getattr(model.base_model, attr_name).bias)
    #         else:
    #             raise NotImplementedError(f"Vocabulary Patcher supports 'random' initialization strategy only")
    
    return model


def prepare_subprosqa_model_and_tokenizer(model: PreTrainedModel, base_model_name: str, num_concepts: int = 10000) -> tuple:
    """Prepare model and tokenizer for SubProsQA dataset."""
    tokenizer = SubProsQATokenizer(num_concepts=num_concepts)
    model = patch_model(model, tokenizer, initialize_strategy='random')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    return model, tokenizer
