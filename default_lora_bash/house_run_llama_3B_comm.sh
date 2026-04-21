#!/usr/bin/env bash

set -u

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LRS=(2e-4 3e-4 4e-4 5e-4)
EPOCHS=$(seq 10 20)

LOG_DIR="logs/openbookqa_sweep"
mkdir -p "$LOG_DIR"

WANDB_GROUP="openbookqa_lr_epoch_sweep"

run_id=0
total_runs=$(( ${#LRS[@]} * 13 ))

for lr in "${LRS[@]}"; do
  for epoch in $EPOCHS; do
    run_id=$((run_id + 1))

    # wandb 显示名称：保留你的原始描述 + 当前超参数
    wandb_name="full insert | lr=${lr} | epochs=${epoch}"

    # 用于日志文件名，避免空格和特殊字符
    safe_lr=${lr//./p}
    safe_lr=${safe_lr//- /}
    safe_lr=${safe_lr//-/m}
    run_name="full_insert_lr${safe_lr}_ep${epoch}"
    log_file="${LOG_DIR}/${run_name}.log"

    echo "=================================================="
    echo "[$run_id/$total_runs] Start run: $wandb_name"
    echo "log: $log_file"
    echo "=================================================="

    python finetune.py \
        --config-name=finetune \
        wandb.name=""rank8_lr${lr}_ep${epoch}"" \
        wandb.project="lora llama3.2_3b MoLE csqa" \
        model=llama3.2_3b \
        model.lora.r=8 \
        model.lora.lora_nums=1 \
        model.lora.cmole_use_lowrank=false \
        model.lora.modules_to_save='["score"]' \
        model.lora.sparsegen_cfg.enabled=false \
        model.lora.sparsegen_cfg.hidden_sizes=512 \
        gpu=8 \
        scheduler.lr_scheduler_name="CosineAnnealingLR" \
        scheduler.use_warmup=false \
        batch_size=64 \
        val_batch_size=64 \
        dataset=csqa \
        task=csqa \
        task.learning_rate="$lr" \
        ce_loss_coef=1.0 \
        lb_loss_coef=1.0 \
        reg_loss_coef=0.0 \
        gradient_accumulation_steps=1 \
        num_epochs="$epoch" \
        dist_backend=nccl \
        save=False \
        2>&1 | tee "$log_file"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
      echo "[WARN] Run failed: $wandb_name (exit code: $exit_code)"
      echo "[WARN] Continue to next run..."
    else
      echo "[OK] Finished: $wandb_name"
    fi

    echo
  done
done

echo "All runs completed."