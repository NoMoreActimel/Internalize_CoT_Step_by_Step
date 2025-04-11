#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

D=7
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/${D}_by_${D}_mult/gpt2_pretrained/full_cot_from_scratch/"
mkdir -p "$DIR"

export WANDB_API_KEY=""
export WANDB_MODE="offline"

python src/train.py \
    --model gpt2 \
    --train_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/train.txt" \
    --val_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/valid.txt" \
    --epochs 5 \
    --lr 5e-5 \
    --batch_size 64 \
    --accumulate 1 \
    --train_type full-cot \
    --train_from_scratch \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --wandb_project cot-distillation \
    --wandb_run_name fullcot_${D}b${D}_from_scratch \
    --max_new_tokens 512 \
    --save_period 1 \
    --save_model "$DIR" \
    > "$DIR/log_masks.train" 2>&1
