#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

D=7
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/${D}_by_${D}_mult/gpt2_pretrained/jepa_completely_random_masks_from_fullcot1_2/"
mkdir -p "$DIR"

export WANDB_API_KEY=""
export WANDB_MODE="offline"

python src/train.py \
    --model gpt2 \
    --train_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/train.txt" \
    --val_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/valid.txt" \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 64 \
    --accumulate 1 \
    --mode train \
    --train_from_scratch \
    --removal_type random-masks \
    --joint_masked_distribution \
    --replace_mask \
    --train_type jepa-cot-distill \
    --alpha_logits_loss 10.0 \
    --ref_model_update_decay 0.999 \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --wandb_project cot-distillation \
    --wandb_run_name jepa_crandom_masks_${D}b${D}_from_fullcot1 \
    --max_new_tokens 512 \
    --save_period 1 \
    --save_model "$DIR" \
    --from_pretrained_fullcot_checkpoint /scratch/ss19021/Internalize_CoT_Step_by_Step/train_models/7_by_7_mult/gpt2_pretrained/full_cot_from_scratch/checkpoint-epoch0.pth \
    > "$DIR/log_masks.train" 2>&1
