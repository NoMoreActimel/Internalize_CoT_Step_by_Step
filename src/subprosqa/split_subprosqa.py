from sklearn.model_selection import train_test_split
from typing import List, Dict
import json


class DatasetSplitter:
    def __init__(self, dataset_path, split_train_val_test, val_size, test_size, split_random_state):
        self.dataset_path = dataset_path
        self.split_train_val_test = split_train_val_test
        self.val_size = val_size
        self.test_size = test_size
        self.split_random_state = split_random_state

        self.dataset = self._load_dataset(self.dataset_path)

        if self.split_train_val_test:
            self.train, self.val, self.test = self._split_train_val_test()
            dataset_dir, dataset_name = self.dataset_path.rsplit('/', 1)
            for split_name, dataset in zip(['train', 'val', 'test'], [self.train, self.val, self.test]):
                self._save_dataset(dataset, dataset_dir + f'/{split_name}_{dataset_name}')
        self._save_dataset(self.dataset, self.dataset_path)

    def _split_train_val_test(self):
        train_val, test = train_test_split(
            self.dataset, 
            random_state=self.split_random_state,
            test_size=self.test_size
        )
        adjusted_val_size = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            random_state=self.split_random_state,
            test_size=adjusted_val_size
        )
        return train, val, test

    @staticmethod
    def _save_dataset(dataset: List[Dict], dataset_path: str):
        """Save the entire dataset to JSON file."""
        print(f"Saving dataset to {dataset_path}...")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved with {len(dataset)} samples!")
        
    @staticmethod
    def _load_dataset(dataset_path: str) -> List[Dict]:
        """Load the entire dataset from JSON file."""
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        print(f"Dataset loaded with {len(dataset)} samples!")
        return dataset

if __name__ == "__main__":
    DatasetSplitter(
        dataset_path='data/subprosqa/subprosqa_dataset.json',
        split_train_val_test=True,
        val_size=0.1,
        test_size=0.1,
        split_random_state=42
    )
