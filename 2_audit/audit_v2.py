#!/usr/bin/env python
"""HQ-Edit 方向审计 v2：强提示 + 思维链，避免迎合"声称顺序"。
双卡并行，输出 audit2_shard{N}.jsonl（含 reasoning + verdict）。
"""
import argparse
import json
import re

import pyarrow.parquet as pq

PARQUET = "/data/liufeng/qwen-image-edit-sft/data/hqedit_raw/data/HQ-Edit_0.parquet"
TRAIN_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/train"
TEST_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/test"
MODEL_PATH = "/data/liufeng/models/Qwen2.5-VL-7B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
args = parser.parse_args()

items = [("train", i, i) for i in range(141)] + [("test", i, 141 + i) for i in range(20)]
items = items[args.shard::args.num_shards]
edits = pq.read_table(PARQUET, columns=["edit"]).column("edit")

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

PROMPT = (
    'You are a strict QA auditor for an instruction-based image editing dataset. '
    'Do NOT trust any claimed ordering; judge only from actual pixel content.\n'
    'Editing instruction: "{edit}"\n'
    'You see Image A (first) and Image B (second). Follow these steps:\n'
    '1) In Image A, note the key objects/features relevant to the instruction.\n'
    '2) In Image B, do the same.\n'
    '3) Decide the ordering:\n'
    '   - FORWARD means B is the result of applying the instruction to A.\n'
    '   - REVERSED means A already looks like the RESULT and B looks like the SOURCE. '
    'Strong clues of REVERSED: instruction ADDS X but A already contains X while B lacks it; '
    'instruction REMOVES X but A lacks X while B still has it.\n'
    '   - UNCLEAR if you cannot determine.\n'
    'Finish with a final line exactly: "VERDICT: FORWARD" or "VERDICT: REVERSED" or "VERDICT: UNCLEAR".'
)

out_path = f"/home/liufeng/projects/qwen-image-edit-sft/2_audit/outputs/audit2_shard{args.shard}.jsonl"
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
            {"type": "text", "text": PROMPT.format(edit=edit)},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        m = re.search(r"VERDICT:\s*(FORWARD|REVERSED|UNCLEAR)", raw.upper())
        verdict = m.group(1) if m else "UNCLEAR"
        fout.write(json.dumps({"split": split, "idx": i, "row": r,
                               "verdict": verdict, "reason": raw, "edit": edit},
                              ensure_ascii=False) + "\n")
        fout.flush()
        done += 1
        print(f"[s{args.shard}] {done}/{len(items)} {split}#{i:04d} -> {verdict}", flush=True)
print(f"[s{args.shard}] DONE")
