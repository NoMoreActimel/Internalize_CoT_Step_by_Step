
#!/bin/bash

set -euo pipefail

# Move to your project root
cd /home/ss19021/Internalize_CoT_Step_by_Step/

# Scratch/dataset locations
SCRATCH="/scratch/ss19021/Internalize_CoT_Step_by_Step"
OUTDIR="${SCRATCH}/data/openmath"

mkdir -p "$OUTDIR"

python prepare_openmath.py \
  --path                    open-r1/OpenR1-Math-220k \
  --name                    default \
  --split                   train \
  --split_train_val_test \
  --val_size                0.1 \
  --test_size               0.1 \
  --split_random_state      42 \
  --filter_max_str_length   1024 \
  --filter_key              answer \
  --output_dir              "$OUTDIR" \
  > "$OUTDIR/prepare.log" 2>&1

echo "Done. Files written to $OUTDIR/{train.txt,valid.txt,test.txt}"
