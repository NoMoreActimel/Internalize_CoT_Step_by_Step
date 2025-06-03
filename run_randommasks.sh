#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

D=9
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/${D}_by_${D}_mult/gpt2_pretrained/completely_random_masks_joint_from_scratch_replace/"
mkdir -p "$DIR"

python src/train.py \
    --model gpt2 \
    --train_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/train.txt" \
    --val_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/valid.txt" \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 64 \
    --accumulate 1 \
    --removal_type random-masks \
    --joint_masked_distribution \
    --replace_mask \
    --train_from_scratch \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --wandb_project cot-distillation \
    --wandb_run_name joint_${D}b${D}_random_masks_from_scratch_labels_replace \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_masks.train" 2>&1
