from dataclasses import dataclass
import os
import copy
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import get_sep_position, extract_answer, extract_cot


def get_chunks_positions(sample, tokenizer, chunk_size):
    if len(sample.shape) == 1:
        sample = sample.unsqueeze(0)

    first_sep_positions = get_sep_position(sample, tokenizer.eos_token_id)
    second_sep_positions = get_sep_position(sample, tokenizer.eos_token_id, skip=1)

    left_border, right_border = first_sep_positions.item() + 1, second_sep_positions.item()

    chunk_positions = list(range(left_border, right_border, chunk_size))
    return chunk_positions

def assign_new_tokens_to_chunks(chunk_positions, new_token_ids):
    chunk_new_token_ids = list(random.choices(new_token_ids, k=len(chunk_positions)))
    return chunk_new_token_ids

def add_new_tokens(model, tokenizer, num_new_tokens):
    orig_vocab_length = len(tokenizer.get_vocab())

    new_tokens = [f"<ULTRA_MEGA_NEW_TOKEN_{i}" for i in range(num_new_tokens)]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    vocab_length = len(tokenizer.get_vocab())

    assert num_added_tokens == num_new_tokens, "New tokens addition failed, change the new token template?"
    assert orig_vocab_length + num_new_tokens == vocab_length, "New tokens addition failed, change the new token template?"

    model.base_model.resize_token_embeddings(len(tokenizer))
    new_token_ids = [id for id in range(orig_vocab_length, vocab_length)]
    return model, tokenizer, new_token_ids


class CoTDatasetChunks(Dataset):
    def __init__(
            self,
            tokenizer,
            path,
            max_length=-1,
            max_size=-1,
            chunk_size=8,
            num_new_tokens=1000,
            new_token_ids=None,
            pad_cot=False,
            max_cot_length=-1,
            cot_pad_id=None
    ):
        """
            This dataset precomputes chunk positions for each of the samples in the file_path,
            assigning randomly sampled new_token_ids for each chunk.
        """
        super().__init__()

        assert os.path.isfile(path), f"Input file path {path} not found"
        print (f'Creating features from dataset file at {path}')

        self.tokenizer, self.new_token_ids = tokenizer, new_token_ids
        
        self.eos_tok = self.tokenizer.eos_token
        self.separator = tokenizer.eos_token_id

        self.file_path = path
        self.max_length = max_length
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.num_new_tokens = num_new_tokens

        self.pad_cot = pad_cot
        self.max_cot_length = max_cot_length
        self.cot_pad_id = cot_pad_id
        if self.cot_pad_id is None:
            self.cot_pad_id = self.tokenizer.encode("<cot>", add_special_tokens=False)[0]
            print(f'CoT Pad Id was not provided, using {self.cot_pad_id} corresponding to "<cot>"!')

        lines = self._read_lines()
        self.dataset = self._process_examples(lines)
        if self.pad_cot:
            emp_max_cot_length = self._compute_max_cot_length()
            if self.max_cot_length is not None:
                if emp_max_cot_length > self.max_cot_length:
                    print(f"""
                          [WARNING] Specified max_cot_length = {self.max_cot_length} 
                          is smaller than observed max CoT length = {emp_max_cot_length},
                          switching to it!!!
                    """)
                    self.max_cot_length = emp_max_cot_length
            else:
                self.max_cot_length = emp_max_cot_length
            self._pad_cots()
        

    def _read_lines(self):
        with open(self.file_path, encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                if (len(line.strip()) > 0 and not line.strip().isspace() and len(line.strip().split('||')) == 2):
                    lines.append(line.strip().split('||'))
                    if self.max_size > 0 and len(lines) >= self.max_size:
                        break

        return lines

    def _compute_max_cot_length(self):
        max_len = 0
        for item in self.dataset:
            seps = (item["input_ids"] == self.separator).nonzero(as_tuple=True)[0]
            if len(seps) >= 2:
                cot_len = seps[1].item() - seps[0].item() - 1
                max_len = max(cot_len, max_len)
        print(f"Empirical max CoT length is {max_len}!")
        return max_len

    def _pad_cots(self):
        print(f"Padding CoTs with pad token: {self.cot_pad_id} up to max length: {self.max_cot_length}!")
        for item in self.dataset:
            seps = (item["input_ids"] == self.separator).nonzero(as_tuple=True)[0]
            cot_length = seps[1].item() - seps[0].item() - 1
            n_cot_pad_ids = self.max_cot_length - cot_length

            if n_cot_pad_ids > 0:
                item["input_ids"] = torch.cat([
                    item["input_ids"][:seps[1].item()],
                    torch.full((n_cot_pad_ids,), self.cot_pad_id, dtype=item["input_ids"].dtype),
                    item["input_ids"][seps[1].item():]
                ], dim=0)
        
    def _process_examples(self, lines):
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        examples_all = []
        for src, tgt in zip(src_lines, tgt_lines):
            ans = extract_answer(tgt)
            cot = extract_cot(tgt)

            sent = f' {src} {self.eos_tok} {cot} {self.eos_tok} {ans} {self.eos_tok} '

            sent_encoded = self.tokenizer(
                [sent],
                add_special_tokens=True,
                truncation=(self.max_length > 0),
                max_length=self.max_length if self.max_length > 0 else None,
                return_tensors="pt"
            )

            input_ids = sent_encoded["input_ids"][0]
            if self.chunk_size:
                chunk_positions = get_chunks_positions(input_ids, self.tokenizer, self.chunk_size)
            if self.num_new_tokens:
                chunk_new_token_ids = assign_new_tokens_to_chunks(chunk_positions, self.new_token_ids)

            examples_all.append({
                "input_ids": input_ids,
                "chunk_positions": chunk_positions if self.chunk_size else None,
                "chunk_input_ids": chunk_new_token_ids if self.num_new_tokens else None
            })

            if len(examples_all) % 10000 == 0:
                print (len(examples_all))
        
        return examples_all
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        input_ids = item["input_ids"]

        labels = copy.deepcopy(input_ids)
        # sep_idx = labels.index(self.separator) + 1
        sep_idx = (labels == self.separator).int().argmax().item() + 1
        labels[:sep_idx] = -100

        return {
            "input_ids": torch.tensor(input_ids.clone().detach(), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "chunk_input_ids": torch.tensor(item["chunk_input_ids"], dtype=torch.long) if self.num_new_tokens else None,
            "chunk_positions": item["chunk_positions"] if self.chunk_size else None
        }

@dataclass
class CoTDataCollatorChunks:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        input_ids = [item["input_ids"] for item in items]
        labels = [item["labels"] for item in items]
        chunk_input_ids = [item["chunk_input_ids"] for item in items]
        chunk_positions = [item["chunk_positions"] for item in items]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        if chunk_input_ids[0] is not None:
            chunk_input_ids = pad_sequence(chunk_input_ids, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'chunk_input_ids': chunk_input_ids,
            'chunk_positions': chunk_positions
        }
