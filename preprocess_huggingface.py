import argparse
import json
import numpy as np
import os

from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
# from sklearn.model_selection import train_test_split


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
            filter_key="answer",
            json_dataset=False,
            **kwargs
            ):
        super().__init__(
            path, name, streaming,
            split, data_files, max_samples,
            shuffle, shuffle_seed,
            **kwargs
        )

        self._filter_correct()
        self._preprocess_format()
        
        if filter_max_str_length is not None:
            self.filter_length(filter_min_str_length, filter_max_str_length, filter_key)

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
    
    def filter_length(self, min_str_length, max_str_length, key):
        dataset_size = len(self.dataset)
        new_dataset = []
        for sample in self.dataset:
            if (max_str_length is None) or (len(sample[key]) <= max_str_length):
                if (min_str_length is None) or (len(sample[key]) >= min_str_length):
                    new_dataset.append(sample)
        self.dataset = new_dataset

        print(f' \
            Filtered dataset by key="{key}" string min_length="{min_str_length} and max_length="{max_str_length}", \
            size reduced from {dataset_size} to {len(self.dataset)} samples! \
        ')
    
    # def _train_val_test_split(self):
    #     train_plus_val, test = train_test_split(
    #         self.dataset,
    #         test_size=self.test_size,
    #         random_state=self.split_random_state
    #     )
    #     # Now split train+val into TRAIN and VAL
    #     # we want val_size fraction *of the original*, so relative val on train_plus_val is:
    #     rel_val_size = self.val_size / (1.0 - self.test_size)
    #     train, val = train_test_split(
    #         train_plus_val,
    #         test_size=rel_val_size,
    #         random_state=self.split_random_state
    #     )

    #     print(f"Split data into: train: {len(train)}, val: {len(val)}, test: {len(test)}")
    #     return train, val, test
    
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
                        f.write(f'{item["question"]}||{item["answer"]}')
            print(f"Wrote formatted dataset into {path}!")


class OpenR1MathDataset(HuggingFacePreprocessDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mean_length = 0
        max_length = 0
        for i, sample in enumerate(self.dataset):
            # if i % 1000 == 0:
                # print(i, len(sample['answer']), sample)
            mean_length += len(sample['answer'])
            max_length = max(max_length, len(sample['answer']))

        print("-" * 30)
        print("Final dataset size:", len(self.dataset))
        print("Mean length:", mean_length / len(self.dataset))
        print("Max length:", max_length)
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

        print(f' \
            Filtered dataset by correctness and completeness of reasoning", \
            size reduced from {dataset_size} to {len(self.dataset)} samples! \
        ')
    

class OpenMathInstructDataset(HuggingFacePreprocessDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mean_length = 0
        max_length = 0
        for i, sample in enumerate(self.dataset):
            mean_length += len(sample['answer'])
            max_length = max(max_length, len(sample['answer']))

        print("-" * 30)
        print("Final dataset size:", len(self.dataset))
        print("Mean length:", mean_length / len(self.dataset))
        print("Max length:", max_length)
        print("-" * 60)
    
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

    # filtering
    parser.add_argument('--filter_max_str_length', type=int, default=None, help="drop samples whose `--filter_key` is longer")
    parser.add_argument('--filter_min_str_length', type=int, default=None, help="drop samples whose `--filter_key` is shorter")
    parser.add_argument('--filter_key', type=str, default="answer", help="which field to check length against")

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
        "filter_key":           args.filter_key,
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
