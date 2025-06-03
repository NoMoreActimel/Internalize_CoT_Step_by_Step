#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/gsm8k/ll3b_pretrained/ll3b_randommasks_052925/"
mkdir -p "$DIR"

which python
which accelerate

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=${SLURM_GPUS_ON_NODE} \
  --num_machines=1 \
src/train.py \
    --mode train \
    --model meta-llama/Llama-3.2-3B \
    --huggingface_dataset \
    --path openai/gsm8k \
    --name main \
    --train_split train \
    --val_split test \
    --test_split test \
    --epochs 32 \
    --lr 5e-5 \
    --batch_size 1 \
    --accumulate 32 \
    --n_generative_eval_batches 64 \
    --train_type cot-distill \
    --removal_type random-masks \
    --joint_masked_distribution \
    --replace_mask \
    --pretrain_epochs 0 \
    --save_period 3 \
    --seed 3456 \
    --reset_optimizer \
    --wandb_project cot-distillation \
    --wandb_run_name gsm8k_ll3b_randommasks \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_masks.train" 2>&1

# --from_pretrained_checkpoint /scratch/ss19021/Internalize_CoT_Step_by_Step/train_models/gsm8k/ll3b_pretrained/ll3b_randommasks_2/checkpoint-epoch3.pth \
