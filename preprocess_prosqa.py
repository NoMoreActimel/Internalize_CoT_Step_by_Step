import argparse
import json
import os

def convert_json_to_txt(root_dir, output_dir=None):
    """
    For each .json file in root_dir, load it (expecting a list of dicts with
    keys "question", "steps", "answer"), and write out a .txt file with the same
    base name. Each line in the .txt is:
      question || steps joined by newlines + " #### answer"
    """
    if output_dir is None:
        output_dir = root_dir
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(root_dir):
        if not fname.endswith('.json'):
            continue
        in_path = os.path.join(root_dir, fname)
        out_fname = os.path.splitext(fname)[0] + '.txt'
        out_path = os.path.join(output_dir, out_fname)

        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(out_path, 'w', encoding='utf-8') as out:
            for sample in data:
                steps = sample.get('steps', [])
                question = sample['question']
                cot = ' '.join(steps) + f" #### {sample['answer']}"
                out.write(f"{question}||{cot}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=".", required=True, help="Root directory that contains .json files, to be processed into .txt")
    parser.add_argument('--output_dir', type=str, default=None, required=False, help="Output directory for processed .txt files, defaults to root_dir")
    args = parser.parse_args()
    convert_json_to_txt(args.root_dir, args.output_dir)
