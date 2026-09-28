#!/usr/bin/env python
"""用 Qwen2.5-VL 评估基线编辑质量：给定 源图 + 生成图 + 编辑指令，判断编辑是否正确执行。

输出 judge_baseline.jsonl，每行: index / score(0-2) / reasoning
运行: CUDA_VISIBLE_DEVICES=2 /data/liufeng/conda_envs/sft/bin/python judge_baseline.py
（sft 环境含 Qwen2.5-VL 依赖，与之前 audit 脚本一致）
"""
import json

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/data/liufeng/models/Qwen2.5-VL-7B-Instruct"
SUMMARY = "/data/liufeng/qwen-image-edit-sft/results/baseline/summary.json"
OUT = "/data/liufeng/qwen-image-edit-sft/results/baseline/judge_baseline.jsonl"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

with open(SUMMARY, "r", encoding="utf-8") as f:
    summary = json.load(f)

PROMPT = (
    'You are judging an instruction-based image editing model.\n'
    'Editing instruction: "{edit}"\n'
    'Image 1 is the SOURCE (before). Image 2 is the MODEL OUTPUT (after).\n'
    'Answer two questions:\n'
    '1) EDIT EXECUTION: Did the model correctly execute the instruction in Image 2? '
    'Score: 2 = fully executed, 1 = partially executed or executed with artifacts, 0 = not executed / wrong edit.\n'
    '2) PRESERVATION: Are the unedited regions (background, other objects, composition) preserved? '
    'Score: 2 = well preserved, 1 = some unintended changes, 0 = severely distorted or unrelated output.\n'
    'Respond in exactly this format (one line each):\n'
    'EDIT: <0|1|2>\nPRESERVE: <0|1|2>\nNOTE: <one short sentence>'
)

with open(OUT, "w", encoding="utf-8") as fout:
    for item in summary:
        i = item["index"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{item['source']}"},
            {"type": "image", "image": f"file://{item['generated']}"},
            {"type": "text", "text": PROMPT.format(edit=item["prompt"])},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip()
        edit_score = preserve_score = -1
        note = ""
        for line in resp.splitlines():
            line = line.strip()
            if line.upper().startswith("EDIT:"):
                try: edit_score = int(line.split(":")[1].strip()[0])
                except Exception: pass
            elif line.upper().startswith("PRESERVE:"):
                try: preserve_score = int(line.split(":")[1].strip()[0])
                except Exception: pass
            elif line.upper().startswith("NOTE:"):
                note = line.split(":", 1)[1].strip()
        rec = {"index": i, "prompt": item["prompt"], "edit_score": edit_score,
               "preserve_score": preserve_score, "note": note, "raw": resp}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{i:04d}] edit={edit_score} preserve={preserve_score} | {note[:80]}")

print(f"\n完成 -> {OUT}")
