#!/usr/bin/env python
"""验证 HQ-Edit 训练数据方向是否正确。

背景: 用户发现 train_src_0000.png(源图A) 带现代要素、train_dst_0000.png(目标图B) 反而像初始图，
怀疑数据方向反了。而已知证据(官方字段定义 + parquet 文本)都指向 input_image=编辑前、output_image=编辑后。

Part 1: 像素级核对 —— 将 parquet 原始 input_image/output_image 与已保存的 *_src_*.png/*_dst_*.png
        逐一比对，排除转换脚本把图存反的可能。
Part 2: 内容判断 —— 用本地 Qwen2.5-VL-7B 描述这些图的实际内容(是否含全息显示屏/悬浮平台等
        未来要素)，与 parquet 文本(input=传统市场 / output=未来市场)对照，给出最终结论。

运行: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
      /data/liufeng/conda_envs/qwen25vl/bin/python -u verify_direction.py
"""
import gc
import hashlib
import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PARQUET = "/data/liufeng/qwen-image-edit-sft/data/hqedit_raw/data/HQ-Edit_0.parquet"
TRAIN_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/train"
TEST_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/test"
MODEL_PATH = "/data/liufeng/models/Qwen2.5-VL-7B-Instruct"

# (split, 目录, parquet行号范围)  train 第 i 条 = parquet 行 i; test 第 i 条 = parquet 行 141+i
CHECK_ROWS = [
    ("train", TRAIN_DIR, 0, 3),
    ("test", TEST_DIR, 141, 142),
]


def load_cell(cell):
    """parquet 图片单元格 -> PIL Image (RGB)"""
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    raise ValueError(f"未知图片单元格类型: {type(cell)}")


def pix_hash(img):
    return hashlib.md5(np.asarray(img).tobytes()).hexdigest()[:12]


print("读取 parquet 图片列(需加载全部行, 稍等)...", flush=True)
table = pq.read_table(PARQUET, columns=["input_image", "output_image", "edit"])
print(f"共 {table.num_rows} 行", flush=True)

print("\n===== Part 1: 像素级核对 (保存的 PNG vs parquet 原图) =====", flush=True)
for split, base_dir, lo, hi in CHECK_ROWS:
    for r in range(lo, hi):
        in_hash = pix_hash(load_cell(table.column("input_image")[r].as_py()))
        out_hash = pix_hash(load_cell(table.column("output_image")[r].as_py()))
        i = r if split == "train" else r - 141
        src_hash = pix_hash(Image.open(f"{base_dir}/{split}_src_{i:04d}.png").convert("RGB"))
        dst_hash = pix_hash(Image.open(f"{base_dir}/{split}_dst_{i:04d}.png").convert("RGB"))
        print(f"[{split} #{i:04d}] src==input_image: {'一致' if src_hash == in_hash else '不一致!!'}"
              f" | dst==output_image: {'一致' if dst_hash == out_hash else '不一致!!'}", flush=True)

del table
gc.collect()

print("\n===== Part 2: Qwen2.5-VL 判断图片实际内容 =====", flush=True)
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

QUESTION = ("Describe this image in one sentence. Then answer on a new line: does it contain any "
            "futuristic elements such as holographic displays or floating platforms? Answer YES or NO.")


def ask(path):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": path},
        {"type": "text", "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=100)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


# 重新只读文本列(轻量)
edits = pq.read_table(PARQUET, columns=["edit"]).column("edit")
for split, base_dir, lo, hi in CHECK_ROWS:
    for r in range(lo, hi):
        i = r if split == "train" else r - 141
        print(f"\n--- [{split} #{i:04d}] edit 指令: {edits[r].as_py()}", flush=True)
        for tag, path in [
            ("src (edit_image/源图A)", f"{base_dir}/{split}_src_{i:04d}.png"),
            ("dst (image/目标图B)", f"{base_dir}/{split}_dst_{i:04d}.png"),
        ]:
            print(f"  {tag} => {ask(path)}", flush=True)

print("\n===== 验证完成 =====", flush=True)
