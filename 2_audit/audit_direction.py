#!/usr/bin/env python
"""全量审计 HQ-Edit 子集(161条)的图片方向是否与 edit 指令一致。

对每条样本同时给 VLM 看 src(声称的编辑前) 和 dst(声称的编辑后)，
判定: FORWARD(src->dst 正确) / REVERSED(图片反了) / UNCLEAR(无法判断)。

双卡并行:
  CUDA_VISIBLE_DEVICES=2 ... audit_direction.py --shard 0 --num_shards 2
  CUDA_VISIBLE_DEVICES=3 ... audit_direction.py --shard 1 --num_shards 2
输出: audit_shard{N}.jsonl (实时flush)
"""
import argparse
import json
import os

import pyarrow.parquet as pq

PARQUET = "/data/liufeng/qwen-image-edit-sft/data/hqedit_raw/data/HQ-Edit_0.parquet"
TRAIN_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/train"
TEST_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/test"
MODEL_PATH = "/data/liufeng/models/Qwen2.5-VL-7B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
args = parser.parse_args()

# 全局样本列表: (split, 文件序号i, parquet行号r)
items = []
for i in range(141):
    items.append(("train", i, i))
for i in range(20):
    items.append(("test", i, 141 + i))
items = items[args.shard::args.num_shards]

edits = pq.read_table(PARQUET, columns=["edit"]).column("edit")

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

PROMPT_TMPL = (
    'You are auditing an instruction-based image editing dataset. '
    'Image 1 is claimed as the INPUT (before editing), Image 2 as the OUTPUT (after editing). '
    'Editing instruction: "{edit}" '
    'According to the actual image contents, reply with exactly one word: '
    'FORWARD if applying the instruction to Image 1 yields Image 2; '
    'REVERSED if applying it to Image 2 would yield Image 1 (pair is swapped); '
    'UNCLEAR if neither or you cannot tell.'
)

out_path = f"/home/liufeng/projects/qwen-image-edit-sft/2_audit/outputs/audit_shard{args.shard}.jsonl"
done = 0
with open(out_path, "w", encoding="utf-8") as fout:
    for split, i, r in items:
        base = TRAIN_DIR if split == "train" else TEST_DIR
        src = f"{base}/{split}_src_{i:04d}.png"
        dst = f"{base}/{split}_dst_{i:04d}.png"
        edit = str(edits[r].as_py())
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{src}"},
            {"type": "image", "image": f"file://{dst}"},
            {"type": "text", "text": PROMPT_TMPL.format(edit=edit)},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        verdict = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)[0].strip().upper()
        if "REVERS" in verdict:
            verdict = "REVERSED"
        elif "FORWARD" in verdict:
            verdict = "FORWARD"
        else:
            verdict = "UNCLEAR"
        rec = {"split": split, "idx": i, "row": r, "verdict": verdict, "edit": edit}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        done += 1
        print(f"[shard{args.shard}] {done}/{len(items)} {split}#{i:04d} -> {verdict}", flush=True)

print(f"[shard{args.shard}] DONE -> {out_path}", flush=True)
