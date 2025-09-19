import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset

from src.dataset.data_chunked import CoTDatasetChunks
from src.utils import COT_ANSWER_SPLIT_PATTERN


class CoTChunksHFDataset(CoTDatasetChunks, Dataset):
    def __init__(
            self,
            tokenizer,
            path,
            name=None,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None,
            question_key="question",
            answer_key="answer",
            max_length=-1,
            max_size=-1,
            chunk_size=8,
            num_new_tokens=1000,
            new_token_ids=None,
            **kwargs
    ):
        Dataset.__init__(self)

        self.tokenizer, self.new_token_ids = tokenizer, new_token_ids

        # Flag consistency between HugginFace and chunked datasets
        if max_samples is None and max_size != -1:
            max_samples = max_size
        if max_samples is not None and max_size == -1:
            max_size = max_samples
        
        # in case we have test split only, split it into val/test if mentioned
        if split is not None and ";" in split:
            split, manual_split = split.split(";") # must be e.g. "test;val" or "test;test"
        else:
            manual_split = None

        # proper datasets.load_dataset call:
        if max_samples is None:
            self.dataset = load_dataset(path, name=name, split=split, data_files=data_files)
        else:
            assert split is not None
            dataset = load_dataset(path, name=name, split=split, data_files=data_files)
            if shuffle:
                dataset = dataset.shuffle(shuffle_seed) if shuffle_seed is not None else dataset.shuffle()
            self.dataset = dataset.select(range(max_samples))
        
        self.question_key = question_key
        self.answer_key = answer_key
        
        self.eos_tok = self.tokenizer.eos_token
        self.separator = tokenizer.eos_token_id

        self.max_length = max_length
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.num_new_tokens = num_new_tokens

        self._init_pad_attributes(**kwargs)

        lines = self._read_lines()
        lines = self._format_answers(lines)
        self.dataset = self._process_examples(lines)
        
        self._pad_if_needed() # <- we compute the same max-length for train / val / test

        # we need to split after preprocessing all samples each time for same max-length padding
        if manual_split: 
            split_seed = shuffle_seed if shuffle_seed is not None else 1
            self.split_manually(manual_split, split_seed=split_seed)

    def split_manually(self, manual_split, split_seed):
        if manual_split not in ("val", "test"):
            raise ValueError(f"Manual split must be one of 'val' or 'test', found {manual_split}")

        rng = np.random.default_rng(split_seed)
        indices = np.arange(len(self.dataset))
        rng.shuffle(indices)

        half = len(self.dataset) // 2
        if manual_split == "val":
            selected_indices = indices[:half]
        else:  # manual_split == "test"
            selected_indices = indices[half:]

        self.dataset = [self.dataset[int(i)] for i in selected_indices]
    
    def _format_answers(self, lines):
        new_lines = []
        for q, a in lines:
            al, ar = a.split(COT_ANSWER_SPLIT_PATTERN.strip())
            al, ar = al.strip(), ar.strip()
            a = al + COT_ANSWER_SPLIT_PATTERN + ar
            new_lines.append((q, a))
        return new_lines
    
    def _read_lines(self):
        return [(item[self.question_key], item[self.answer_key]) for item in self.dataset]
