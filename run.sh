# Use physical GPUs 4,5,6,7. Inside this process they are remapped to cuda:0,1,2,3.
# Disable Weights & Biases logging for this run.
# WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py \
    --config-name=finetune \
    wandb.name="full insert 2e-4" \
    wandb.project="llama3.2_3b MoLE hellaswag" \
    model=llama3.2_3b \
    model.lora.modules_to_save='["score"]' \
    model.lora.sparsegen_cfg.enabled=false \
    model.lora.sparsegen_cfg.hidden_sizes=512 \
    gpu=4 \
    scheduler.lr_scheduler_name="CosineAnnealingLR" \
    scheduler.use_warmup=false \
    batch_size=8 \
    val_batch_size=8 \
    dataset=hellaswag \
    task=hellaswag \
    task.learning_rate=3e-4 \
    ce_loss_coef=1.0 \
    lb_loss_coef=1.0 \
    reg_loss_coef=0.0 \
    gradient_accumulation_steps=8 \
    num_epochs=10 \
    dist_backend=nccl 
