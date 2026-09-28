#!/bin/bash
# 多卡数据并行推理：每张卡跑一个进程，各处理一部分测试集
# 用法: NUM_SHARDS=3 bash infer_parallel.sh <output_dir> [lora_path]
#   NUM_SHARDS=3 bash infer_parallel.sh /data/.../results/lora /data/.../epoch-1.safetensors
#   NUM_SHARDS=1 bash infer_parallel.sh /data/.../results/baseline   # 不带 LoRA
#
# 注意:
#   - 每个分片进程需在 CPU 内存常驻一份 float8 权重 (~31GB)，
#     4 分片约 124GB 会逼近 125GB 物理内存上限，默认 3 分片 (~93GB) 更安全
#   - DIFFSYNTH_MODEL_BASE_PATH 必须设置，否则回退 modelscope 在线下载，
#     多进程并发下载会因文件锁竞争死锁（已踩坑）

OUT_DIR=$1
LORA_PATH=$2
NUM_SHARDS=${NUM_SHARDS:-3}
PYTHON=/home/liufeng/conda_envs/diffsynth/bin/python
SCRIPT_DIR=/home/liufeng/projects/qwen-image-edit-sft

# 使用本地模型权重，禁止任何在线下载
export DIFFSYNTH_MODEL_BASE_PATH=/data/liufeng/models
export HF_HUB_OFFLINE=1

if [ -z "$OUT_DIR" ]; then
    echo "用法: NUM_SHARDS=3 bash infer_parallel.sh <output_dir> [lora_path]"
    exit 1
fi

mkdir -p "$OUT_DIR"

for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$SHARD $PYTHON $SCRIPT_DIR/infer_testset.py \
        --output_dir "$OUT_DIR" \
        --num_shards $NUM_SHARDS --shard $SHARD \
        ${LORA_PATH:+--lora "$LORA_PATH"} \
        > "$OUT_DIR/shard_$SHARD.log" 2>&1 &
    echo "已启动分片 $SHARD (GPU $SHARD), pid=$!"
    sleep 20   # 错峰加载，避免同时读盘峰值
done

echo "$NUM_SHARDS 个分片已全部启动，日志: $OUT_DIR/shard_*.log"
echo "查看进度: ls $OUT_DIR/test_*.png | wc -l   (目标 20)"
wait
echo "全部完成"
