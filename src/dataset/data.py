from torch.utils.data import DataLoader

from src.dataset.data_stepbystep import CoTDataset, CoTDataCollator
from src.dataset.data_random_cot import CoTDatasetRandomCot, CoTDataCollatorRandomCot
from src.dataset.data_chunked import CoTDatasetChunks, CoTDataCollatorChunks
from src.dataset.data_huggingface import CoTChunksHFDataset

def get_data_classes(args, tokenizer, new_token_ids=None, split="train"):
    dataset_kwargs = {}

    if getattr(args, 'huggingface_dataset', None):
        # assert getattr(args, 'removal_type', None) == 'random-chunks' or getattr(args, 'removal_type', None) == 'random-masks'
        dataset_kwargs["path"] = getattr(args, 'path', None)
        dataset_kwargs["name"] = getattr(args, 'name', None)
        dataset_kwargs["split"] = split
        dataset_kwargs["data_files"] = getattr(args, 'data_files', None)
        dataset_kwargs["max_samples"] = getattr(args, 'max_samples', None)
        dataset_kwargs["shuffle"] = getattr(args, 'shuffle', None)
        dataset_kwargs["shuffle_seed"] = getattr(args, 'shuffle_seed', None)
        dataset_kwargs["question_key"] = getattr(args, 'question_key', None)
        dataset_kwargs["answer_key"] = getattr(args, 'answer_key', None)
    else:
        dataset_kwargs["path"] = getattr(args, f"{split}_path", None)
    
    if ";" in split:
        split = split.split(";")[1]
    split_max_size = getattr(args, f"{split}_max_size", None)
    default_max_size = getattr(args, 'max_size', None) if split == "train" else -1
    dataset_kwargs["max_size"] = default_max_size if split_max_size is None else split_max_size

    dataset_kwargs["max_length"] = getattr(args, 'max_len_train', None)

    dataset_kwargs["pad_cot"] = getattr(args, 'pad_cot', None)
    dataset_kwargs["max_cot_length"] = getattr(args, 'max_cot_length', None)
    dataset_kwargs["cot_pad_id"] = getattr(args, 'cot_pad_id', None)

    dataset_kwargs["pad_query"] = getattr(args, 'pad_query', None)
    dataset_kwargs["max_query_length"] = getattr(args, 'max_query_length', None)
    dataset_kwargs["query_pad_id"] = getattr(args, 'query_pad_id', None)

    dataset_kwargs["json_dataset"] = getattr(args, 'json_dataset', None)
    dataset_kwargs["chunk_size"] = getattr(args, 'chunk_size', None) if getattr(args, 'removal_type', None) == 'random-chunks' else None
    dataset_kwargs["num_new_tokens"] = getattr(args, 'num_new_tokens', None) if getattr(args, 'removal_type', None) == 'random-chunks' else 0
    dataset_kwargs["new_token_ids"] = new_token_ids if getattr(args, 'removal_type', None) == 'random-chunks' else None

    dataset_kwargs["random_cot_strategy"] = getattr(args, 'random_cot_strategy', None)
    dataset_kwargs["random_cot_length"] = getattr(args, 'random_cot_length', None)

    # In case of huggingface dataset, always use our CoTChunksHFDataset instead of native iCoT one
    ours_sbs_flag = (getattr(args, 'removal_type', None) == 'step-by-step') \
        and (getattr(args, 'huggingface_dataset', None) \
        or getattr(args, 'json_dataset', None))

    if getattr(args, 'random_cot', None):
        DatasetClass = CoTDatasetRandomCot
        CollateClass = CoTDataCollatorRandomCot
    elif getattr(args, 'removal_type', None) == 'step-by-step' and not ours_sbs_flag:
        DatasetClass = CoTDataset
        CollateClass = CoTDataCollator
    elif getattr(args, 'removal_type', None) == 'random-chunks' or getattr(args, 'removal_type', None) == 'random-masks' or ours_sbs_flag:
        DatasetClass = CoTChunksHFDataset if getattr(args, 'huggingface_dataset', None) else CoTDatasetChunks
        CollateClass = CoTDataCollatorChunks
    else:
        raise ValueError(f'''
            args.removal_type must be either "step-by-step", "random-chunks", "random-masks",
            found {getattr(args, "removal_type", None)}
        ''')
    
    dataset = DatasetClass(tokenizer, **dataset_kwargs)
    collate = CollateClass(tokenizer)
    return dataset, collate


def load_data(args, tokenizer, new_token_ids=None):
    train_shuffle = not getattr(args, "disable_train_shuffle", False)
    train_dataset, collate_fn = get_data_classes(args, tokenizer, new_token_ids, split=args.train_split)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=train_shuffle)

    val_dataset, collate_fn = get_data_classes(args, tokenizer, new_token_ids, split=args.val_split)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    if args.test_path or (args.huggingface_dataset and args.test_split and args.test_split != args.val_split):
        if args.test_path and args.test_split is None:
            args.test_split = "test"
        test_dataset, collate_fn = get_data_classes(args, tokenizer, new_token_ids, split=args.test_split)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    return train_dataloader, val_dataloader, None
