#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/gsm8k/gpt2_pretrained/random_contiguous_replace_from_scratch/"
mkdir -p "$DIR"

python src/train.py \
    --model gpt2 \
    --huggingface_dataset \
    --path openai/gsm8k \
    --name main \
    --split train \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 32 \
    --accumulate 1 \
    --removal_type random-masks \
    --joint_masked_distribution \
    --random_contiguous_removal \
    --replace_mask \
    --train_from_scratch \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --wandb_project cot-distillation \
    --wandb_run_name gsm8k_contiguous_replace_from_scratch \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_masks.train" 2>&1
