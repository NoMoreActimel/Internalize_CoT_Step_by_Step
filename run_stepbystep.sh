#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/4_by_4_mult/gpt2_pretrained/sbs_from_scratch/"
mkdir -p "$DIR"

export WANDB_API_KEY="96b7c9ce4fa58a9b8254a7e3b14ef24071ecd75e"
export WANDB_MODE="online"

python src/train.py \
    --model gpt2 \
    --train_path "$SCRATCH_DIR/data/4_by_4_mult/train.txt" \
    --val_path "$SCRATCH_DIR/data/4_by_4_mult/valid.txt" \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 64 \
    --accumulate 1 \
    --train_from_scratch \
    --removal_type step-by-step \
    --remove_per_epoch 8 \
    --remove_all_when_remove_beyond inf \
    --removal_smoothing_lambda 4 \
    --removal_side left \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --save_model "$DIR" \
    --wandb_project cot-distillation \
    --wandb_run_name sbs_4b4_from_scratch \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_stepbystep.train" 2>&1
