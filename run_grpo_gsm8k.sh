#!/bin/bash
export PYTHONPATH="/home/ss19021/Internalize_CoT_Step_by_Step:$PYTHONPATH"

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/gsm8k/ll1b_pretrained/ll1b_grpo_randommasks_082725/"
mkdir -p "$DIR"

which python
which accelerate

# Point to the SFT randommasks checkpoint for this model
SFT_CKPT="$SCRATCH_DIR/train_models/gsm8k/ll1b_pretrained/ll1b_randommasks_032726/model_best.pth"

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
    --best_val_metric "full-cot_accuracy" \
    --epochs 96 \
    --lr 5e-6 \
    --batch_size 1 \
    --accumulate 8 \
    --max_grad_norm 0.1 \
    --n_generative_eval_batches 128 \
    --intermediate_eval \
    --train_type grpo \
    --removal_type random-masks \
    --joint_masked_distribution \
    --replace_mask \
    --prompt_in_percentage \
    --pretrain_epochs 0 \
    --eval_period 3 \
    --save_period 30 \
    --seed 3456 \
    --reset_optimizer \
    --from_pretrained_checkpoint "$SFT_CKPT" \
    --grpo_group_size 16 \
    --grpo_clip_eps 0.2 \
    --grpo_kl_coeff 0.01 \
    --grpo_temperature 1.0 \
    --grpo_inner_steps 4 \
    --wandb_project cot-distillation \
    --wandb_run_name b2_gsm8k_grpo_randommasks_ll1b \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_grpo.train" 2>&1
