from torch.utils.data import DataLoader

from data_stepbystep import CoTDataset, CoTDataCollator
from data_chunked import CoTDatasetChunks, CoTDataCollatorChunks
from data_huggingface import CoTChunksHFDataset

def get_data_classes(args, tokenizer, new_token_ids=None, split="train"):
    dataset_kwargs = {}

    if args.huggingface_dataset:
        assert args.removal_type == 'random-chunks' or args.removal_type == 'random-masks'
        dataset_kwargs["path"] = args.path
        dataset_kwargs["name"] = args.name
        dataset_kwargs["split"] = split
        dataset_kwargs["data_files"] = args.data_files
        dataset_kwargs["max_samples"] = args.max_samples
        dataset_kwargs["shuffle"] = args.shuffle
        dataset_kwargs["shuffle_seed"] = args.shuffle_seed
        dataset_kwargs["question_key"] = args.question_key
        dataset_kwargs["answer_key"] = args.answer_key
    else:
        dataset_kwargs["path"] = getattr(args, f"{split}_path")
    
    dataset_kwargs["max_size"] = args.max_size if split == "train" else -1
    dataset_kwargs["max_length"] = args.max_len_train

    dataset_kwargs["pad_cot"] = args.pad_cot
    dataset_kwargs["max_cot_length"] = args.max_cot_length
    dataset_kwargs["cot_pad_id"] = args.cot_pad_id

    if args.removal_type == 'step-by-step':
        DatasetClass = CoTDataset
        CollateClass = CoTDataCollator
    elif args.removal_type == 'random-chunks' or args.removal_type == 'random-masks':
        DatasetClass = CoTChunksHFDataset if args.huggingface_dataset else CoTDatasetChunks
        CollateClass = CoTDataCollatorChunks

        dataset_kwargs["chunk_size"] = args.chunk_size if args.removal_type == 'random-chunks' else None
        dataset_kwargs["num_new_tokens"] = args.num_new_tokens if args.removal_type == 'random-chunks' else 0
        dataset_kwargs["new_token_ids"] = new_token_ids if args.removal_type == 'random-chunks' else None
    else:
        raise ValueError(f'args.removal_type must be either "step-by-step", "random-chunks", "random-masks", found {args.removal_type}')
    
    dataset = DatasetClass(tokenizer, **dataset_kwargs)
    collate = CollateClass(tokenizer)
    return dataset, collate


def load_data(args, tokenizer, new_token_ids=None):
    train_dataset, collate_fn = get_data_classes(args, tokenizer, new_token_ids, split=args.train_split)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    val_dataset, collate_fn = get_data_classes(args, tokenizer, new_token_ids, split=args.val_split)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    if args.test_path:
        test_dataset, collate_fn = get_data_classes(args, tokenizer, new_token_ids, split=args.test_split)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    return train_dataloader, val_dataloader, None
