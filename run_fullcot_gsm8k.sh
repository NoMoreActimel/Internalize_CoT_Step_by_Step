#!/bin/bash
export PYTHONPATH="/home/ss19021/Internalize_CoT_Step_by_Step:$PYTHONPATH"

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

DATE="041526"
SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/gsm8k/ll1b_pretrained/fullcot_ll1b_${DATE}/"
mkdir -p "$DIR"

which python
which accelerate

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=${SLURM_GPUS_ON_NODE} \
  --num_machines=1 \
src/train.py \
    --mode train \
    --model meta-llama/Llama-3.2-1B \
    --huggingface_dataset \
    --path openai/gsm8k \
    --name main \
    --train_split train \
    --val_split "test;val" \
    --test_split "test;test" \
    --epochs 96 \
    --lr 5e-5 \
    --batch_size 8 \
    --accumulate 16 \
    --n_generative_eval_batches 64 \
    --train_type full-cot \
    --removal_type random-masks \
    --pretrain_epochs 0 \
    --eval_period 3 \
    --seed 3456 \
    --reset_optimizer \
    --save_period 30 \
    --wandb_project cot-distillation \
    --wandb_run_name b8_FULLCOT_gsm8k_ll1b \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_fullcot.train" 2>&1
