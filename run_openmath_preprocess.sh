
#!/bin/bash

set -euo pipefail

# Move to your project root
cd /home/ss19021/Internalize_CoT_Step_by_Step/

# Scratch/dataset locations
SCRATCH="/scratch/ss19021/Internalize_CoT_Step_by_Step"
OUTDIR="${SCRATCH}/data/openmath-instruct-2-augGSM8k"

mkdir -p "$OUTDIR"

# open-r1/OpenR1-Math-220k
# nvidia/OpenMathInstruct-2

python preprocess_huggingface.py \
  --path                    nvidia/OpenMathInstruct-2 \
  --name                    default \
  --split                   train \
  --split_train_val_test \
  --json_dataset \
  --val_size                0.1 \
  --test_size               0.1 \
  --split_random_state      42 \
  --tokenizer_model         "meta-llama/Llama-3.2-1B" \
  --filter_column_name      "problem_source" \
  --filter_column_values    "augmented_gsm8k" \
  --output_dir              "$OUTDIR" \
  > "$OUTDIR/prepare.log" 2>&1

echo "Done. Files written to $OUTDIR/{train.json, valid.json, test.json}"
