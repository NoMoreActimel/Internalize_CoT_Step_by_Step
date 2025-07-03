import argparse
import os

from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split


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
                    dataset = dataset.shuffle(shuffle_seed)
                else:
                    dataset.shuffle()
            
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
            filter_key="answer",
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
            self.filter_max_length(filter_max_str_length, filter_key)

        self.split_train_val_test = split_train_val_test
        self.val_size = val_size
        self.test_size = test_size
        self.split_random_state = split_random_state
        
        if self.split_train_val_test:
            self.train, self.val, self.test = self._train_val_test_split()

    def _preprocess_format(self):
        pass

    def _filter_correct(self):
        pass
    
    def filter_max_length(self, max_str_length, key):
        dataset_size = len(self.dataset)
        new_dataset = []
        for sample in self.dataset:
            if len(sample[key]) <= max_str_length:
                new_dataset.append(sample)
        self.dataset = new_dataset

        print(f' \
            Filtered dataset by key="{key}" sequence max_length="{max_str_length}", \
            size reduced from {dataset_size} to {len(self.dataset)} samples! \
        ')
        # new_dataset_dict = {
        #     key: [sample[key] for sample in new_dataset]
        #     for key in new_dataset[0].keys()
        # }
        # self.dataset = HFDataset.from_dict(new_dataset_dict)
    
    def _train_val_test_split(self):
        train_plus_val, test = train_test_split(
            self.dataset,
            test_size=self.test_size,
            random_state=self.split_random_state
        )
        # Now split train+val into TRAIN and VAL
        # we want val_size fraction *of the original*, so relative val on train_plus_val is:
        rel_val_size = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(
            train_plus_val,
            test_size=rel_val_size,
            random_state=self.split_random_state
        )

        print(f"Split data into: train: {len(train)}, val: {len(val)}, test: {len(test)}")
        return train, val, test



class OpenMathDataset(HuggingFacePreprocessDataset):
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
    
    def write_lines(self, output_dir):
        if self.split_train_val_test:
            datasets = [self.train, self.val, self.test]
            paths = [f"{output_dir}/train.txt", f"{output_dir}/valid.txt", f"{output_dir}/test.txt"]
        else:
            datasets = [self.dataset]
            paths = [f"{output_dir}/train.txt"]
        
        os.makedirs(output_dir, exist_ok=True)
        for dataset, path in zip(datasets, paths):
            with open(path, 'w') as f:
                for item in dataset:
                    f.write(f'{item["question"]}||{item["answer"]}')
            print(f"Wrote formatted dataset into {path}!")


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

    # train/test‐split args
    parser.add_argument('--split_train_val_test', action='store_true', help="perform train / val / test split via sklearn")
    parser.add_argument('--val_size', type=float, default=0.1, help="proportion for test set when splitting")
    parser.add_argument('--test_size', type=float, default=0.1, help="proportion for test set when splitting")
    parser.add_argument('--split_random_state', type=int, default=42)

    # filtering
    parser.add_argument('--filter_max_str_length', type=int, default=None, help="drop samples whose `--filter_key` is longer")
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
        "filter_key":           args.filter_key,
    }

    ds = OpenMathDataset(**ds_kwargs)
    ds.write_lines(args.output_dir)


if __name__ == "__main__":
    main()
