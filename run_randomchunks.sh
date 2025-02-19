#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit
DIR="train_models/4_by_4_mult/gpt2_pretrained/random_chunks_onetoken_from_scratch/"
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
    --removal_type random-chunks \
    --num_new_tokens 1 \
    --chunk_size 8 \
    --remove_by_schedule \
    --removal_side left \
    --removal_schedule "(0,0) (2,1) (4,2) (6,3) (8,4) (10,5) (12,6)" \
    --pretrain_epochs 0 \
    --seed 3456 \
    --reset_optimizer \
    --save_model "$DIR" \
    --wandb_project cot-distillation \
    --wandb_run_name onetoken_4by4_sbs_chunks_from_scratch \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_chunks.train" 2>&1
