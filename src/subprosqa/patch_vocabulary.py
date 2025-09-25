import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)


class SubProsQATokenizer(PreTrainedTokenizer):
    """Custom tokenizer for SubProsQA dataset with graph concepts."""
    
    def __init__(self, num_concepts: int, max_len: int = None):
        super().__init__(max_len=max_len)

        self.__token_ids = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 1,
            '<eos>': 2,
            '<sep>': 3,
            '<mask>': 4,    
            '[EDGES]': 5,    # Edge list marker
            '[FACTS]': 6,    # Facts marker (for hybrid mode)
            'Question:': 7,  # Question marker
            'Answer:': 8,    # Answer marker
            '####': 9,       # Separator in answer format
            '->': 10,         # Edge direction token
            ',': 11,          # Comma separator
            'is': 12,         # "is" relation
            'or': 13,         # "or" in questions
            '?': 14,          # Question mark
            ' ': 15,          # Space token (if treating as separate)
        }
        num_special_tokens = len(self.__token_ids)
        for i in range(self.num_concepts):
            self.__token_ids[f"concept{i:04d}"] = i + num_special_tokens
        
        self.__id_tokens = {value: key for key, value in self.__token_ids.items()}        
        self.__sorted_tokens = sorted(self.__token_ids.keys(), key=len, reverse=True)
        self.max_length = len(self.__sorted_tokens[0])
        self.min_length = len(self.__sorted_tokens[-1])

        self.special_tokens_map = {
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'sep_token': '<sep>',
            'mask_token': '<mask>'
        }
        self.add_special_tokens(self.special_tokens_map)

    def _tokenize(self, text: str, **kwargs):
        """
        Greedy tokenization that matches the longest possible tokens from the vocabulary.
        Raises an error if any part of the text cannot be tokenized.
        """
        tokens = []
        position = 0
        
        while position < len(text):
            matched = False
            
            for token_length in range(self.max_length, self.min_length - 1, -1):
                if text[position:position + token_length] in self.__token_ids:
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
                        f"Cannot tokenize text at position {position}. Text: '{text}'. "
                        f"No matching token found in vocabulary."
                    )
        
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens.get(index, self.unk_token)
    
    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs) -> str:
        """
        Decode token IDs back to text, handling -100 values properly.
        -100 values are ignored during decoding as they represent masked tokens.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Filter out -100 values (masked tokens) and convert to tokens
        valid_token_ids = [token_id for token_id in token_ids if token_id != -100]
        
        if not valid_token_ids:
            return ""

        tokens = []
        for token_id in valid_token_ids:
            if token_id in self.__id_tokens:
                token = self.__id_tokens[token_id]
                if not skip_special_tokens or token not in self.special_tokens_map.values():
                    tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return ''.join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory) / (filename_prefix + 'vocab.json')
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vocab_path, 'w') as f:
            json.dump(self.__token_ids, f, indent=2)
        
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)


def patch_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    initialize_strategy: str = 'random'
) -> PreTrainedModel:
    """Patch model to work with custom tokenizer vocabulary."""
    
    old_vocab_size = model.config.vocab_size
    new_vocab_size = tokenizer.vocab_size
    
    print(f"Resizing model embeddings from {old_vocab_size} to {new_vocab_size}")
    model.resize_token_embeddings(new_vocab_size)
    
    model.config.vocab_size = new_vocab_size
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    for attr_name in ['lm_head', 'score']:
        if hasattr(model, attr_name) and getattr(model, attr_name) is not None:
            print(f"Resizing {attr_name} layer")       
            setattr(model, attr_name, torch.nn.Linear(
                in_features=getattr(model, attr_name).in_features,
                out_features=new_vocab_size,
                bias=getattr(model, attr_name).bias is not None
            ))
            
            if initialize_strategy == 'random':
                torch.nn.init.normal_(getattr(model, attr_name).weight, mean=0.0, std=0.02)
                if getattr(model, attr_name).bias is not None:
                    torch.nn.init.zeros_(getattr(model, attr_name).bias)
            else:
                raise NotImplementedError(f"Vocabulary Patcher supports 'random' initialization strategy only")
    
    return model


def prepare_subprosqa_model_and_tokenizer(model: PreTrainedModel, base_model_name: str, num_concepts: int = 10000) -> tuple:
    """Prepare model and tokenizer for SubProsQA dataset."""
    tokenizer = SubProsQATokenizer(num_concepts=num_concepts)
    model = patch_model(model, tokenizer, initialize_strategy='random')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    return model, tokenizer
