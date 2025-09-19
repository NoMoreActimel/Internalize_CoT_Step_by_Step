#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

D=11
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/${D}_by_${D}_mult/gpt2_pretrained/sbs_from_scratch_4_072225/"
mkdir -p "$DIR"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --deepspeed_plugin ds_config.json \

python src/train.py \
    --model gpt2 \
    --train_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/train.txt" \
    --val_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/valid.txt" \
    --test_path "$SCRATCH_DIR/data/${D}_by_${D}_mult/test_bigbench.txt" \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 32 \
    --n_generative_eval_batches 1 \
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
    --wandb_run_name sbs_${D}b${D}_from_scratch_4 \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    --from_pretrained_checkpoint /scratch/ss19021/Internalize_CoT_Step_by_Step/train_models/11_by_11_mult/gpt2_pretrained/sbs_from_scratch_3_071725/checkpoint-epoch17.pth \
    > "$DIR/log_stepbystep.train" 2>&1
