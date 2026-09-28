#!/usr/bin/env python
"""下载 HQ-Edit 数据集子集（只下载1个parquet文件，提取220条样本）"""

import json
import os
import pyarrow.parquet as pq
from modelscope.hub.file_download import dataset_file_download

# 输出目录
DATA_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 只下载第一个 parquet 文件（约525MB，含约330条数据）
print("下载 HQ-Edit 第一个 parquet 文件...")
parquet_path = dataset_file_download(
    "AI-ModelScope/HQ-Edit",
    file_path="data/HQ-Edit_0.parquet",
    local_dir="/data/liufeng/qwen-image-edit-sft/data/hqedit_raw"
)
print(f"下载完成: {parquet_path}")

# 读取 parquet
print("读取 parquet 文件...")
table = pq.read_table(parquet_path)
df = table.to_pandas()
print(f"该文件包含 {len(df)} 条数据")
print(f"列名: {list(df.columns)}")

NUM_TRAIN = 141
NUM_TEST = 20
TOTAL = NUM_TRAIN + NUM_TEST

# 已知问题行（全量审计确认）：row 0 的 input_image/output_image 与 edit 指令方向相反
# （input_image 是未来市场、output_image 是传统市场，但 edit 是 "Add futuristic elements..."），
# 交换后使用。161 条中仅此一条反转，其余全部正常。
SWAP_INPUT_OUTPUT_ROWS = {0}

def convert_and_save(df, start_idx, count, output_dir, split_name):
    """将 HQ-Edit 数据转换为 DiffSynth 格式"""
    from PIL import Image
    import io

    metadata = []
    for i in range(count):
        idx = start_idx + i
        if idx >= len(df):
            break

        row = df.iloc[idx]

        # HQ-Edit 字段: input_image, output_image, edit
        edit_instruction = str(row.get("edit", ""))
        input_desc = str(row.get("input", ""))
        output_desc = str(row.get("output", ""))

        # 获取图片（可能是 bytes 或 PIL Image）
        src_img_data = row.get("input_image")
        dst_img_data = row.get("output_image")

        # 处理图片数据
        def to_pil_image(data):
            if isinstance(data, dict) and "bytes" in data:
                return Image.open(io.BytesIO(data["bytes"]))
            elif isinstance(data, dict) and "path" in data:
                return Image.open(data["path"])
            elif isinstance(data, (bytes, bytearray)):
                return Image.open(io.BytesIO(data))
            elif hasattr(data, 'save'):  # already PIL
                return data
            else:
                print(f"  未知图片格式: {type(data)}")
                return None

        src_img = to_pil_image(src_img_data)
        dst_img = to_pil_image(dst_img_data)

        # 修复已知方向反转的行
        if idx in SWAP_INPUT_OUTPUT_ROWS:
            src_img, dst_img = dst_img, src_img
            print(f"  [fix] row {idx}: 已交换 input/output（已知方向反转行）")

        if src_img is None or dst_img is None:
            print(f"  跳过第 {idx} 条：图片解析失败")
            continue

        src_name = f"{split_name}_src_{i:04d}.png"
        dst_name = f"{split_name}_dst_{i:04d}.png"
        src_path = os.path.join(output_dir, src_name)
        dst_path = os.path.join(output_dir, dst_name)

        src_img.save(src_path)
        dst_img.save(dst_path)

        metadata.append({
            "image": dst_path,          # 目标图 B
            "edit_image": src_path,      # 源图 A
            "prompt": edit_instruction,  # 编辑指令
        })

        if (i + 1) % 50 == 0:
            print(f"  {split_name}: 已处理 {i+1}/{count}")

    # 写 metadata.json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"{split_name} 完成: {len(metadata)} 条 -> {meta_path}")
    return metadata

# 训练集
print(f"\n=== 处理训练集 ({NUM_TRAIN} 条) ===")
train_meta = convert_and_save(df, 0, NUM_TRAIN, TRAIN_DIR, "train")

# 测试集
print(f"\n=== 处理测试集 ({NUM_TEST} 条) ===")
test_meta = convert_and_save(df, NUM_TRAIN, NUM_TEST, TEST_DIR, "test")

print(f"\n=== 汇总 ===")
print(f"训练集: {len(train_meta)} 条 -> {TRAIN_DIR}")
print(f"测试集: {len(test_meta)} 条 -> {TEST_DIR}")
print(f"\n示例训练数据:")
print(json.dumps(train_meta[0], ensure_ascii=False, indent=2))
