# Prerprocessing of ProntoQA from COCONUT json format, then into txt

import argparse
import json
import os
import subprocess
import sys
import shutil

def preprocess_prontoqa(dataset_path, output_dir, num_train=9000, num_val=200):
    file = json.load(open(dataset_path, "r"))
    data_dict = []

    for k, v in file.items():
        example = v["test_example"]
        data_dict.append(
            {
                "question": example["question"] + " " + example["query"],
                "steps": [
                    " ".join(example["chain_of_thought"][i : i + 2])
                    for i in range(0, len(example["chain_of_thought"]), 2)
                ],
                "answer": example["answer"],
            }
        )
    
    data = []
    for sample in data_dict:
        steps = sample.get('steps', [])
        question = sample['question']
        cot = ' '.join(steps) + f" #### {sample['answer']}"
        data.append(f"{question}||{cot}\n")
    os.makedirs(output_dir, exist_ok=True)
    
    bounds = [(0, num_train), (num_train, num_train+num_val), (num_train+num_val, len(data))]
    for phase, (l, r) in zip(["train", "valid", "test"], bounds):
        # out_fname = dataset_path.split('/')[-1].split('.')[0] + f"_{phase}.txt"
        out_fname = f"prontoqa_{phase}.txt"
        out_path = os.path.join(output_dir, out_fname)
        with open(out_path, "w") as out:
            for i, sample in enumerate(data):
                if i >= l and i < r:
                    out.write(sample)
                
        print(f"Wrote preprocessed dataset to {out_path}!")

def generate_prontoqa(repo_path, output_path, num_trials=10000, min_hops=5, max_hops=5):
    cmd = [
        sys.executable,
        os.path.join(repo_path, "run_experiment.py"),
        "--model-name", "json",
        "--model-size", "0",
        "--ordering", "random",
        "--num-trials", str(num_trials),
        "--few-shot-examples", "0",
        "--ontology", "fictional",
        "--min-hops", str(min_hops),
        "--max-hops", str(max_hops),
        "--hops-skip", "1",
    ]
    subprocess.run(cmd, cwd=repo_path, check=True)
    if min_hops < max_hops:
        output_path = output_path.rsplit(".", 1)[0]
    for hop in range(min_hops, max_hops + 1):
        filename = f"{hop}hop_0shot_random.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = output_path
        if out[-5:] != ".json":
            out = out + f"_{hop}hop.json"
        shutil.move(f"{repo_path}/{filename}", out)
        print(f"Filename {filename} is done")
    
    print(f"Generated ProntoQA dataset into {output_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_dataset_path', type=str, default=".", required=True, help="ProntoQA dataset path in orig format")
    parser.add_argument('--generate_dataset', action='store_true', default=False, help="Generate dataset with original code into ")
    parser.add_argument('--prontoqa_repo_path', type=str, default=None, required=False, help="Path to oficial code, only for generation")
    parser.add_argument('--num_trials', type=int, default=10000)
    parser.add_argument('--num_train', type=int, default=9000)
    parser.add_argument('--num_val', type=int, default=200)
    parser.add_argument('--min_hops', type=int, default=5)
    parser.add_argument('--max_hops', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None, required=True, help="Output directory for processed .txt files")
    args = parser.parse_args()
    
    if args.generate_dataset:
        if os.path.exists(args.orig_dataset_path):
            print(f"File {args.orig_dataset_path} already exists; skipping ProntoQA generation.")
        else:
            generate_prontoqa(args.prontoqa_repo_path, args.orig_dataset_path, args.num_trials, args.min_hops, args.max_hops)
    
    assert args.min_hops == args.max_hops, "Generation works for different min/max hops, preprocessing not yet, fix code with multiple datasets"
    preprocess_prontoqa(args.orig_dataset_path, args.output_dir, args.num_train, args.num_val)
