import argparse
import json
import numpy as np
import os

from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
# from sklearn.model_selection import train_test_split

try:
    from transformers import AutoTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class HuggingFaceDataset(TorchDataset):
    def __init__(
            self,
            path,
            name=None,
            streaming=False,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None,
            **kwargs
            ):
        super().__init__()

        if max_samples is not None:
            assert not streaming
            assert split is not None

            dataset = load_dataset(path, name=name, split=split, data_files=data_files)
            if shuffle:
                if shuffle_seed is not None:
                    dataset = dataset.shuffle(seed=shuffle_seed)
                else:
                    dataset = dataset.shuffle()
            
            self.dataset = dataset.select(range(max_samples))
        else:
            self.dataset = load_dataset(
                path,
                name=name,
                streaming=streaming,
                split=split,
                data_files=data_files
            )
        
        self.streaming=streaming
    
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        if self.streaming:
            return int(float('inf'))
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class HuggingFacePreprocessDataset(HuggingFaceDataset):
    def __init__(
            self,
            path,
            name=None,
            streaming=False,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None,
            split_train_val_test=True,
            val_size=0.1,
            test_size=0.1,
            split_random_state=42,
            filter_max_str_length=None,
            filter_min_str_length=None,
            filter_max_tokens=None,
            filter_min_tokens=None,
            filter_key="answer",
            tokenizer_model=None,
            json_dataset=False,
            **kwargs
            ):
        super().__init__(
            path, name, streaming,
            split, data_files, max_samples,
            shuffle, shuffle_seed,
            **kwargs
        )

        self.kwargs = kwargs

        # Initialize tokenizer if token filtering is requested
        self.tokenizer = None
        if (filter_max_tokens is not None or filter_min_tokens is not None):
            if not TOKENIZERS_AVAILABLE:
                raise ImportError("transformers library is required for token-based filtering. Install with: pip install transformers")
            if tokenizer_model is None:
                raise ValueError("tokenizer_model must be specified when using token-based filtering")
            
            print(f"Loading tokenizer: {tokenizer_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        self._filter_correct()
        self._preprocess_format()
        
        # Apply string length filtering
        if filter_max_str_length is not None or filter_min_str_length is not None:
            self.filter_length(filter_min_str_length, filter_max_str_length, filter_key, use_tokens=False)
        
        # Apply token length filtering
        if filter_max_tokens is not None or filter_min_tokens is not None:
            self.filter_length(filter_min_tokens, filter_max_tokens, filter_key, use_tokens=True)

        self.split_train_val_test = split_train_val_test
        self.val_size = val_size
        self.test_size = test_size
        self.split_random_state = split_random_state

        self.json_dataset = json_dataset
        
        if self.split_train_val_test:
            self.train, self.val, self.test = self._train_val_test_split()

    def _preprocess_format(self):
        pass

    def _filter_correct(self):
        pass
    
    def filter_length(self, min_length, max_length, key, use_tokens=False):
        dataset_size = len(self.dataset)
        new_dataset = []
        
        length_type = "tokens" if use_tokens else "characters"
        print(f"Filtering by {length_type} for key='{key}' with min={min_length}, max={max_length}")
       

        keys = key.split(',') if ',' in key else [key]
        for sample in self.dataset:
            text = '\n'.join([f"{k}: {sample[k]}" for k in keys])
    
            if use_tokens:
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not initialized for token-based filtering")
                length = len(self.tokenizer.encode(text))
            else:
                length = len(text)
            
            # Check if sample passes length filters
            if (max_length is None or length <= max_length) and (min_length is None or length >= min_length):
                new_dataset.append(sample)
        
        self.dataset = new_dataset
        print(f'Filtered dataset by {length_type}, size reduced from {dataset_size} to {len(self.dataset)} samples!')
    
    def _train_val_test_split(self):
        n = len(self.dataset)
        idx = np.random.default_rng(self.split_random_state).permutation(n)

        n_test = int(np.floor(self.test_size * n))
        rel_val = self.val_size / (1 - self.test_size)
        n_val = int(np.floor(rel_val * (n - n_test)))

        def take(ids):
            if hasattr(self.dataset, "select"):
                return self.dataset.select(ids.tolist())
            return [self.dataset[i] for i in ids]

        test = take(idx[:n_test])
        val = take(idx[n_test:n_test + n_val])
        train = take(idx[n_test + n_val:])
        print(f"Split data into: train: {len(train)}, val: {len(val)}, test: {len(test)}")
        return train, val, test

    def write_lines(self, output_dir):
        ext = "json" if self.json_dataset else "txt"
        if self.split_train_val_test:
            datasets = [self.train, self.val, self.test]
            paths = [f"{output_dir}/train.{ext}", f"{output_dir}/valid.{ext}", f"{output_dir}/test.{ext}"]
        else:
            datasets = [self.dataset]
            paths = [f"{output_dir}/train.{ext}"]
        
        os.makedirs(output_dir, exist_ok=True)
        for dataset, path in zip(datasets, paths):
            if self.json_dataset:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, ensure_ascii=False)
            else:
                with open(path, 'w') as f:
                    for item in dataset:
                        f.write(f'{item["question"]}||{item["answer"]}\n')
            print(f"Wrote formatted dataset into {path}!")


class OpenR1MathDataset(HuggingFacePreprocessDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mean_length = 0
        max_length = 0
        mean_tokens = 0
        max_tokens = 0
        
        for i, sample in enumerate(self.dataset):
            # Character-based stats
            char_length = len(sample['answer'])
            mean_length += char_length
            max_length = max(max_length, char_length)
            
            # Token-based stats (if tokenizer available)
            if self.tokenizer is not None:
                token_length = len(self.tokenizer.encode(sample['answer']))
                mean_tokens += token_length
                max_tokens = max(max_tokens, token_length)

        print("-" * 30)
        print("Final dataset size:", len(self.dataset))
        print("Mean character length:", mean_length / len(self.dataset))
        print("Max character length:", max_length)
        
        if self.tokenizer is not None:
            print("Mean token length:", mean_tokens / len(self.dataset))
            print("Max token length:", max_tokens)
            
        print("-" * 60)
    
    def _preprocess_format(self):
        items = []
        for item in self.dataset:
            items.append({
                "question": item['problem'],
                "answer": f"{item['solution']} #### {item['answer']}"
            })
        self.dataset = items

    def _filter_correct(self):
        dataset_size = len(self.dataset)
        items = []
        for item in self.dataset:
            if all(item["correctness_math_verify"]) and all(item["is_reasoning_complete"]):
                items.append(item)
        self.dataset = items

        print(f'Filtered dataset by correctness and completeness of reasoning, '
              f'size reduced from {dataset_size} to {len(self.dataset)} samples!')
    

class OpenMathInstructDataset(HuggingFacePreprocessDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_column_name = self.kwargs.get("filter_column_name", None)
        self.filter_column_values = self.kwargs.get("filter_column_values", None).strip().split(',')
        self._filter_column_name(self.filter_column_name, self.filter_column_values)

        mean_length = 0
        max_length = 0
        mean_tokens = 0
        max_tokens = 0
        
        for i, sample in enumerate(self.dataset):
            # Character-based stats
            char_length = len(sample['answer'])
            mean_length += char_length
            max_length = max(max_length, char_length)
            
            # Token-based stats (if tokenizer available)
            if self.tokenizer is not None:
                token_length = len(self.tokenizer.encode(sample['answer']))
                mean_tokens += token_length
                max_tokens = max(max_tokens, token_length)

        print("-" * 30)
        print("Final dataset size:", len(self.dataset))
        print("Mean character length:", mean_length / len(self.dataset))
        print("Max character length:", max_length)
        
        if self.tokenizer is not None:
            print("Mean token length:", mean_tokens / len(self.dataset))
            print("Max token length:", max_tokens)
            
        print("-" * 60)

    def _filter_column_name(self, filter_column_name, filter_column_values):
        if filter_column_name is None:
            return
        
        dataset_size = len(self.dataset)
        self.dataset = [
            item for item in self.dataset
            if item[filter_column_name] in filter_column_values
        ]
        print(f"Filtered dataset by {filter_column_name} with values {filter_column_values}, "
              f"size reduced from {dataset_size} to {len(self.dataset)} samples!")
    
    def _preprocess_format(self):
        items = []
        for item in self.dataset:
            items.append({
                "question": item['problem'],
                "answer": f"{item['generated_solution']} #### {item['expected_answer']}"
            })
        self.dataset = items


def main():
    parser = argparse.ArgumentParser(
        description="Load, filter, split and dump the OpenMath-style dataset"
    )
    # core HF dataset args
    parser.add_argument('--path', type=str, required=True, help="🤗 dataset path (e.g. 'openmath/openmath')")
    parser.add_argument('--name', type=str, default=None, help="subset name, if any")
    parser.add_argument('--streaming', action='store_true', help="use streaming mode")
    parser.add_argument('--split', type=str, default=None, help="which split to load (train/validation/test)")
    parser.add_argument('--data_files', type=str, default=None, help="JSON or comma-sep list of data_files arg for load_dataset")
    parser.add_argument('--max_samples', type=int, default=None, help="truncate to N samples before filtering")
    parser.add_argument('--shuffle', action='store_true', help="shuffle before truncation")
    parser.add_argument('--shuffle_seed', type=int, default=None)

    parser.add_argument('--json_dataset', action='store_true')

    # train/test‐split args
    parser.add_argument('--split_train_val_test', action='store_true', help="perform train / val / test split via sklearn")
    parser.add_argument('--val_size', type=float, default=0.1, help="proportion for test set when splitting")
    parser.add_argument('--test_size', type=float, default=0.1, help="proportion for test set when splitting")
    parser.add_argument('--split_random_state', type=int, default=42)

    # filtering - string-based
    parser.add_argument('--filter_max_str_length', type=int, default=None, help="drop samples whose `--filter_key` is longer (characters)")
    parser.add_argument('--filter_min_str_length', type=int, default=None, help="drop samples whose `--filter_key` is shorter (characters)")
    
    # filtering - token-based (new)
    parser.add_argument('--filter_max_tokens', type=int, default=None, help="drop samples whose `--filter_key` is longer (tokens)")
    parser.add_argument('--filter_min_tokens', type=int, default=None, help="drop samples whose `--filter_key` is shorter (tokens)")
    parser.add_argument('--tokenizer_model', type=str, default=None, help="model name/path for tokenizer (required for token filtering)")
    
    parser.add_argument('--filter_key', type=str, default="answer", help="which field to check length against")

    parser.add_argument('--filter_column_name', type=str, default=None, help="Key to filter by sample type in OpenMath-Instruct-2")
    parser.add_argument(
        '--filter_column_values', type=str, default=None,
        help="Comma separated values of column, i.e. 'augmented_gsm8k,augmented_gsm8k"
    )

    # output
    parser.add_argument('--output_dir', type=str, default="data/openmath/", required=True, help="where to write formatted datasets")

    args = parser.parse_args()

    ds_kwargs = {
        "path":                 args.path,
        "name":                 args.name,
        "streaming":            args.streaming,
        "split":                args.split,
        "data_files":           args.data_files,
        "max_samples":          args.max_samples,
        "shuffle":              args.shuffle,
        "shuffle_seed":         args.shuffle_seed,
        "split_train_val_test": args.split_train_val_test,
        "val_size":             args.val_size,
        "test_size":            args.test_size,
        "split_random_state":   args.split_random_state,
        "filter_max_str_length":args.filter_max_str_length,
        "filter_min_str_length":args.filter_min_str_length,
        "filter_max_tokens":    args.filter_max_tokens,
        "filter_min_tokens":    args.filter_min_tokens,
        "filter_key":           args.filter_key,
        "filter_column_name":   args.filter_column_name,
        "filter_column_values": args.filter_column_values,
        "tokenizer_model":      args.tokenizer_model,
        "json_dataset":         args.json_dataset
    }

    if args.path == "nvidia/OpenMathInstruct-2":
        ds = OpenMathInstructDataset(**ds_kwargs)
    elif args.path == "open-r1/OpenR1-Math-220k":
        ds = OpenR1MathDataset(**ds_kwargs)
    else:
        raise ValueError
    ds.write_lines(args.output_dir)


if __name__ == "__main__":
    main()
