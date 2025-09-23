import json
import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)

class SubProsQAVocabularyPatcher:
    """Patches tokenizer and model vocabulary for SubProsQA dataset with graph concepts."""
    
    def __init__(self, num_concepts: int = 10000):
        self.num_concepts = num_concepts
        self.vocab = self._create_vocabulary()
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
    
    def _create_vocabulary(self) -> List[str]:
        vocab = []
        
        special_tokens = [
            '<pad>',      # Padding token
            '<unk>',      # Unknown token
            '<bos>',      # Beginning of sequence
            '<eos>',      # End of sequence
            '<sep>',      # Separator token
            '<mask>',     # Mask token for MLM (if needed)
            'Mask',       # Mask token for random masking
            'M'           # Mask token for random masking
        ]
        vocab.extend(special_tokens)
        
        structure_tokens = [
            '[EDGES]',    # Edge list marker
            '[FACTS]',    # Facts marker (for hybrid mode)
            'Question:',  # Question marker
            'Answer:',    # Answer marker
            '####',       # Separator in answer format
            '->',         # Edge direction token
            ',',          # Comma separator
            'is',         # "is" relation
            'or',         # "or" in questions
            '?',          # Question mark
            ' ',          # Space token (if treating as separate)
        ]
        vocab.extend(structure_tokens)
        
        # Concept tokens (concept0000 to conceptNNNN)
        for i in range(self.num_concepts):
            vocab.append(f"concept{i:04d}")
        
        return vocab
    
    def create_custom_tokenizer(self, base_tokenizer_path: str = 'gpt2') -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
        
        tokenizer.vocab = self.token_to_id
        tokenizer.ids_to_tokens = self.id_to_token
        
        tokenizer.pad_token = '<pad>'
        tokenizer.unk_token = '<unk>'
        tokenizer.bos_token = '<bos>'
        tokenizer.eos_token = '<eos>'
        tokenizer.sep_token = '<sep>'
        tokenizer.mask_token = '<mask>'

        tokenizer.pad_token_id = self.token_to_id['<pad>']
        tokenizer.unk_token_id = self.token_to_id['<unk>']
        tokenizer.bos_token_id = self.token_to_id['<bos>']
        tokenizer.eos_token_id = self.token_to_id['<eos>']
        tokenizer.sep_token_id = self.token_to_id['<sep>']
        tokenizer.mask_token_id = self.token_to_id['<mask>']
        
        return tokenizer
    
    def patch_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        initialize_strategy: str = 'random'
    ) -> PreTrainedModel:

        old_vocab_size = model.config.vocab_size
        new_vocab_size = len(tokenizer.vocab)
        
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

def prepare_subprosqa_model_and_tokenizer(model: PreTrainedModel, num_concepts: int = 10000) -> tuple:
    patcher = SubProsQAVocabularyPatcher(num_concepts=num_concepts)
    tokenizer = patcher.create_custom_tokenizer()
    model = patcher.patch_model(model, tokenizer, initialize_strategy='random')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    return model, tokenizer, patcher
