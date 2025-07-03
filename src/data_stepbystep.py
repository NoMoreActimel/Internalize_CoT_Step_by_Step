from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import extract_answer, extract_cot


class CoTDataset(Dataset):
    def __init__(self, tokenizer, path, max_length=-1, max_size=-1, **kwargs):
        assert os.path.isfile(path), f"Input file path {path} not found"
        print (f'Creating features from dataset file at {path}')
        eos_tok = tokenizer.eos_token

        with open(path, encoding="utf-8") as f:
            #lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
            #                                                                 and len(line.split('||')) ==2 )]
            lines = [line.strip().split('||') for line in f.readlines() if (len(line.strip()) > 0 and not line.strip().isspace()
                                                                             and len(line.strip().split('||')) ==2 )]
        if max_size > 0:
            print (f'truncated to {max_size}')
            lines = lines[:max_size]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        #edited_sents_cot = []
        #edited_sents_only = []
        edited_sents_all = []
        #edited_sents_nocot = []
        self.examples_all = []
        for src, tgt in zip(src_lines, tgt_lines):
            #import pdb; pdb.set_trace()
            ans = extract_answer(tgt)
            cot = extract_cot(tgt)
            #sent = ' {} {} '.format(src, eos_tok) + cot + ' {}'.format(eos_tok)
            #edited_sents_cot.append(sent)
            #sent = ' {} {} '.format(src, eos_tok)
            #edited_sents_only.append(sent)
            sent = ' {} {} '.format(src, eos_tok) + cot + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)
            #edited_sents_all.append(sent)
            #sent = ' {} {} '.format(src, eos_tok) + ans + ' {}'.format(eos_tok)
            #edited_sents_nocot.append(sent)
            if max_length > 0:
                batch_encoding_all = tokenizer([sent], add_special_tokens=True, truncation=True, max_length=max_length)
            else:
                batch_encoding_all = tokenizer([sent], add_special_tokens=True)
            self.examples_all.append(batch_encoding_all["input_ids"][0])
            if len(self.examples_all) % 10000 == 0:
                print (len(self.examples_all))

        separator = tokenizer.eos_token_id #tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]
        self.separator = separator

    def __len__(self):
        return len(self.examples_all)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        input_ids = self.examples_all[i]
        labels = copy.deepcopy(input_ids)
        sep_idx = labels.index(self.separator) + 1
        labels[:sep_idx] = [-100] * sep_idx
        return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                )


class CoTDatasetWithoutTokenization(Dataset):
    def __init__(self, path, max_size=-1, **kwargs):
        assert os.path.isfile(path), f"Input file path {path} not found"
        print (f'Creating features from dataset file at {path}')

        with open(path, encoding="utf-8") as f:
            lines = [
                line.strip().split('||')
                for line in f.readlines()
                if (len(line.strip()) > 0 and not line.strip().isspace() and len(line.strip().split('||')) == 2)
            ]
        
        if max_size > 0:
            print (f'truncated to {max_size}')
            lines = lines[:max_size]
        
        src_lines, tgt_lines = list(zip(*lines))
        self.src_lines = list(src_lines)
        self.tgt_lines = list(tgt_lines)

    def __len__(self):
        return len(self.examples_all)

    def __getitem__(self, i):
        return {"src": self.src_lines[i], "tgt": self.tgt_lines[i]}
    

@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_all, labels_all = zip(*examples)
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        labels_all = self._tensorize_batch(labels_all)
        return {'input_ids': input_ids_all, 'labels': labels_all}

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)

@dataclass
class CoTDataCollatorWithTokenization:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer, max_length=-1):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.separator = tokenizer.eos_token_id
        self.max_length = max_length

    def __call__(self, examples):
        input_ids_all, labels_all = zip(*examples)
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        labels_all = self._tensorize_batch(labels_all)
        return {'input_ids': input_ids_all, 'labels': labels_all}

    def _tokenize_batch(self, examples):
        input_ids_all, labels_all = [], []

        for example in examples:
            src, tgt = example["src"], example["tgt"]
            ans = extract_answer(tgt)
            cot = extract_cot(tgt)

            sent = ' {} {} '.format(src, self.eos_token) + cot + ' {} '.format(self.eos_token) + ans + ' {}'.format(self.eos_token)
            
            if self.max_length > 0:
                batch_encoding_all = self.tokenizer([sent], add_special_tokens=True, truncation=True, max_length=self.max_length)
            else:
                batch_encoding_all = self.tokenizer([sent], add_special_tokens=True)

            input_ids = batch_encoding_all["input_ids"][0]
            labels = copy.deepcopy(input_ids)
            sep_idx = labels.index(self.separator) + 1
            labels[:sep_idx] = [-100] * sep_idx
            
            input_ids_all.append(torch.tensor(input_ids, dtype=torch.long))
            labels_all.append(torch.tensor(labels, dtype=torch.long))
        
        return input_ids_all, labels_all

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)



