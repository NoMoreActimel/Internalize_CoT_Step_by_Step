#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

D=7
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/${D}_by_${D}_mult/gpt2_pretrained/sbs_from_scratch_ft_10_04_03_25/"
mkdir -p "$DIR"

export WANDB_API_KEY="96b7c9ce4fa58a9b8254a7e3b14ef24071ecd75e"
export WANDB_MODE="offline"

python src/train.py \
    --model gpt2 \
    --train_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/train.txt" \
    --val_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/valid.txt" \
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
    --seed 3457 \
    --reset_optimizer \
    --save_model "$DIR" \
    --wandb_project cot-distillation \
    --wandb_run_name sbs_${D}b${D}_from_scratch_ft_10 \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    --from_pretrained_checkpoint /scratch/ss19021/Internalize_CoT_Step_by_Step/train_models/7_by_7_mult/gpt2_pretrained/sbs_from_scratch/checkpoint-epoch10.pth \
    > "$DIR/log_stepbystep.train" 2>&1
