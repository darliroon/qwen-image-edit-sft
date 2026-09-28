#!/bin/bash
# 等 baseline 推理完成（20 张图全齐）后自动启动 LoRA 训练
BASELINE_DIR=/data/liufeng/qwen-image-edit-sft/results/baseline
LOG=/data/liufeng/qwen-image-edit-sft/pipeline.log

echo "$(date) 等待 baseline 完成..." >> $LOG
while true; do
    N=$(ls $BASELINE_DIR/test_*.png 2>/dev/null | wc -l)
    echo "$(date) baseline 进度: $N/20" >> $LOG
    if [ "$N" -ge 20 ]; then
        break
    fi
    sleep 60
done

echo "$(date) baseline 完成，等 60 秒释放显存后启动训练..." >> $LOG
sleep 60

echo "$(date) ===== 启动训练 =====" >> $LOG
bash /home/liufeng/projects/qwen-image-edit-sft/3_train/train_lora_hqedit.sh >> $LOG 2>&1
echo "$(date) ===== 训练结束 exit=$? =====" >> $LOG
