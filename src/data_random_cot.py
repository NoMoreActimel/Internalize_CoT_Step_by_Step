from dataclasses import dataclass
import os
import copy
import json
import random
import torch
import re
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import get_sep_position, extract_answer, extract_cot


def extract_digits_from_question(question):
    """Extract all digits from a multiplication question."""
    digits = re.findall(r'\d', question)
    return digits


def generate_random_cot_random_digits(n_tokens):
    """Strategy 1: Completely random digits separated with space."""
    random_digits = [str(random.randint(0, 9)) for _ in range(n_tokens)]
    return ' '.join(random_digits)


def generate_random_cot_random_digits_from_question(question, n_tokens):
    """Strategy 2: Randomly sampled tokens from question, new permutation each sample."""
    digits = extract_digits_from_question(question)
    if not digits:
        # Fallback to strategy 1 if no digits found
        return generate_random_cot_random_digits(n_tokens)
    
    # Sample with replacement to get n_tokens
    sampled_digits = random.choices(digits, k=n_tokens)
    return ' '.join(sampled_digits)


def generate_random_cot_fixed_permutation_random_digits_from_question(question, n_tokens, fixed_permutation):
    """
        Fixed permutation of tokens from question for all samples.
        All samples MUST have the same number of digits in question.
    """
    digits = sorted(extract_digits_from_question(question))
    if not digits:
        return generate_random_cot_random_digits(n_tokens)
    
    return [digits[digit_idx] for digit_idx in fixed_permutation]


class CoTDatasetRandomCot(Dataset):
    def __init__(
            self,
            tokenizer,
            path,
            json_dataset,
            max_length=-1,
            max_size=-1,
            random_cot_strategy="random_digits",
            random_cot_length=50,
            pad_cot=False,
            max_cot_length=-1,
            cot_pad_id=None,
            pad_query=False,
            max_query_length=-1,
            query_pad_id=None,
            **kwargs
    ):
        """
        This dataset replaces CoTs with random CoTs using one of three strategies:
        1. "random_digits":
            Completely random digits separated with space
        2. "random_digits_from_question":
            Randomly sampled tokens from question (digits), new permutation each sample
        3. "fixed_permutation_random_digits_from_question":
            Same as 2, but fixed permutation for all samples.
            All samples MUST have the same number of digits in question.
        """
        super().__init__()

        assert os.path.isfile(path), f"Input file path {path} not found"
        print(f'Creating features from dataset file at {path}')

        self.tokenizer = tokenizer
        
        self.eos_tok = self.tokenizer.eos_token
        self.separator = tokenizer.eos_token_id

        self.json_dataset = json_dataset

        self.file_path = path
        self.max_length = max_length
        self.max_size = max_size
        self.random_cot_strategy = random_cot_strategy
        self.random_cot_length = random_cot_length
        self.fixed_permutation = None

        self._init_pad_attributes(pad_cot, max_cot_length, cot_pad_id, pad_query, max_query_length, query_pad_id)

        lines = self._read_lines()
        self.dataset = self._process_examples(lines)
        
        self._pad_if_needed()
        
    def _init_pad_attributes(self, pad_cot, max_cot_length, cot_pad_id, pad_query, max_query_length, query_pad_id, **kwargs):
        self.pad_cot = pad_cot
        self.max_cot_length = max_cot_length
        self.cot_pad_id = cot_pad_id
        if self.cot_pad_id is None:
            self.cot_pad_id = self.tokenizer.encode("<cot>", add_special_tokens=False)[0]
            print(f'CoT Pad Id was not provided, using {self.cot_pad_id} corresponding to "<cot>"!')

        self.pad_query = pad_query
        self.max_query_length = max_query_length
        self.query_pad_id = query_pad_id
        if self.query_pad_id is None:
            self.query_pad_id = self.tokenizer.encode("<query>", add_special_tokens=False)[0]
            print(f'Query Pad Id was not provided, using {self.query_pad_id} corresponding to "<query>"!')

    def _pad_if_needed(self):
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

        if self.pad_query:
            emp_max_query_length = self._compute_max_query_length()
            if self.max_query_length is not None:
                if emp_max_query_length > self.max_query_length:
                    print(f"""
                          [WARNING] Specified max_query_length = {self.max_query_length} 
                          is smaller than observed max query length = {emp_max_query_length},
                          switching to it!!!
                    """)
                    self.max_query_length = emp_max_query_length
            else:
                self.max_query_length = emp_max_query_length
            self._pad_queries()

    def _read_lines(self):
        if self.json_dataset:
            return self._read_json_lines()
        
        with open(self.file_path, encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                if (len(line.strip()) > 0 and not line.strip().isspace() and len(line.strip().split('||')) == 2):
                    lines.append(line.strip().split('||'))
                    if self.max_size > 0 and len(lines) >= self.max_size:
                        break

        return lines

    def _read_json_lines(self):
        """ Dataset stored as json: list of dicts with "question" and "answer" fields. """
        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON must be a list of dicts.")

        lines = []
        for item in data:
            q = item.get("question")
            a = item.get("answer")
            if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                lines.append((q, a))
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

    def _compute_max_query_length(self):
        max_len = 0
        for item in self.dataset:
            seps = (item["input_ids"] == self.separator).nonzero(as_tuple=True)[0]
            if len(seps) >= 1:
                # Query length is from start to first separator (excluding the separator)
                query_len = seps[0].item()
                max_len = max(query_len, max_len)
        print(f"Empirical max query length is {max_len}!")
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

    def _pad_queries(self):
        print(f"Padding queries with pad token: {self.query_pad_id} up to max length: {self.max_query_length}!")
        for item in self.dataset:
            seps = (item["input_ids"] == self.separator).nonzero(as_tuple=True)[0]
            query_length = seps[0].item()
            n_query_pad_ids = self.max_query_length - query_length

            if n_query_pad_ids > 0:
                item["input_ids"] = torch.cat([
                    item["input_ids"][:seps[0].item()],
                    torch.full((n_query_pad_ids,), self.query_pad_id, dtype=item["input_ids"].dtype),
                    item["input_ids"][seps[0].item():]
                ], dim=0)

    def _generate_random_cot(self, question):
        """Generate random CoT based on the selected strategy."""
        if self.random_cot_strategy == "random_digits":
            return generate_random_cot_random_digits(self.random_cot_length)
        elif self.random_cot_strategy == "random_digits_from_question":
            return generate_random_cot_random_digits_from_question(question, self.random_cot_length)
        elif self.random_cot_strategy == "fixed_permutation_random_digits_from_question":
            if self.fixed_permutation is None:
                digits = sorted(extract_digits_from_question(question))
                self.fixed_permutation = random.choices(list(range(len(digits))), k=self.random_cot_length)
            return generate_random_cot_fixed_permutation_random_digits_from_question(
                question, self.random_cot_length, self.fixed_permutation
            )
        else:
            raise ValueError(f"Invalid random_cot_strategy: {self.random_cot_strategy}.")
        
    def _process_examples(self, lines):
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        examples_all = []
        for src, tgt in zip(src_lines, tgt_lines):
            ans = extract_answer(tgt)
            random_cot = self._generate_random_cot(src)

            sent = f' {src} {self.eos_tok} {random_cot} {self.eos_tok} {ans} {self.eos_tok} '

            sent_encoded = self.tokenizer(
                [sent],
                add_special_tokens=True,
                truncation=(self.max_length > 0),
                max_length=self.max_length if self.max_length > 0 else None,
                return_tensors="pt"
            )

            input_ids = sent_encoded["input_ids"][0]

            examples_all.append({
                "input_ids": input_ids,
            })

            if len(examples_all) % 10000 == 0:
                print(len(examples_all))
        
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
        }

class CoTDataCollatorRandomCot:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        input_ids = [item["input_ids"] for item in items]
        labels = [item["labels"] for item in items]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {'input_ids': input_ids, 'labels': labels,}
