#!/usr/bin/env python
"""把 train 141 对 src|dst 拼成 contact sheet，每对下方标注 edit 指令，供逐张视觉审计。"""
import json
import math
import os
import textwrap

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

TRAIN_DIR = "/data/liufeng/qwen-image-edit-sft/data/hqedit/train"
PARQUET = "/data/liufeng/qwen-image-edit-sft/data/hqedit_raw/data/HQ-Edit_0.parquet"
OUT_DIR = "/home/liufeng/projects/qwen-image-edit-sft/2_audit/sheets"
os.makedirs(OUT_DIR, exist_ok=True)

THUMB = 300
BAND = 118
COLS = 2
ROWS_PER = 4          # 每张 8 对
N = 141

edits = pq.read_table(PARQUET, columns=["edit"]).column("edit")

try:
    FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    FT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except Exception:
    FB = FT = ImageFont.load_default()


def thumb(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((THUMB, THUMB), Image.LANCZOS)


n_sheets = math.ceil(N / (COLS * ROWS_PER))
for sh in range(n_sheets):
    W = COLS * 2 * THUMB
    H = ROWS_PER * (THUMB + BAND)
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for k in range(COLS * ROWS_PER):
        idx = sh * COLS * ROWS_PER + k
        if idx >= N:
            break
        x = (k % COLS) * 2 * THUMB
        y = (k // COLS) * (THUMB + BAND)
        sheet.paste(thumb(f"{TRAIN_DIR}/train_src_{idx:04d}.png"), (x, y))
        sheet.paste(thumb(f"{TRAIN_DIR}/train_dst_{idx:04d}.png"), (x + THUMB, y))
        draw.rectangle([x + 5, y + 5, x + 150, y + 46], fill=(200, 0, 0))
        draw.text((x + 14, y + 6), f"#{idx:04d}", fill=(255, 255, 255), font=FB)
        draw.line([(x + 2 * THUMB - 2, y), (x + 2 * THUMB - 2, y + THUMB)], fill=(0, 220, 0), width=4)
        # 标签
        draw.text((x + 6, y + THUMB - 30), "SRC", fill=(0, 255, 255), font=FB)
        draw.text((x + THUMB + 6, y + THUMB - 30), "DST", fill=(255, 200, 0), font=FB)
        # edit 文本
        edit = str(edits[idx].as_py())
        wrapped = textwrap.fill(edit, width=70)
        draw.multiline_text((x + 6, y + THUMB + 6), wrapped, fill=(230, 230, 230), font=FT, spacing=2)
    out = f"{OUT_DIR}/sheet_{sh:02d}.png"
    sheet.save(out)
    print(f"{out}  idx {sh*COLS*ROWS_PER:04d}-{min((sh+1)*COLS*ROWS_PER-1, N-1):04d}", flush=True)
print("DONE")
