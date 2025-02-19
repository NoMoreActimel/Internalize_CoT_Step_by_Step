#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit
DIR="train_models/4_by_4_mult/gpt2_pretrained/random_masks_joint_from_scratch/"
mkdir -p "$DIR"

python src/train.py \
    --model gpt2 \
    --train_path data/4_by_4_mult/train.txt \
    --val_path data/4_by_4_mult/valid.txt \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 64 \
    --accumulate 1 \
    --train_from_scratch \
    --removal_type random-masks \
    --joint_masked_distribution \
    --left_to_right_removal \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --save_model "$DIR" \
    --wandb_project cot-distillation \
    --wandb_run_name joint_4b4_masks \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_masks.train" 2>&1
