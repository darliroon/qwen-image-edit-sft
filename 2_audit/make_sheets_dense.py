#!/usr/bin/env python
"""密集缩略图总览：每张 sheet 24 对(src|dst)，无文字，供肉眼快速扫明显反转。
共 6 张(train 141) + 1 张(test 20)。"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = "/data/liufeng/qwen-image-edit-sft/data/hqedit"
OUT_DIR = "/home/liufeng/projects/qwen-image-edit-sft/2_audit/sheets"
os.makedirs(OUT_DIR, exist_ok=True)

TH = 200
PAIR_W = 2 * TH
COLS = 3          # 每行 3 对
ROWS = 8          # 每列 8 行 => 24 对/张

try:
    FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
except Exception:
    FB = ImageFont.load_default()


def thumb(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((TH, TH), Image.LANCZOS)


def build(split, n):
    per = COLS * ROWS
    n_sheets = math.ceil(n / per)
    for sh in range(n_sheets):
        sheet = Image.new("RGB", (COLS * PAIR_W, ROWS * TH), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for k in range(per):
            idx = sh * per + k
            if idx >= n:
                break
            base = f"{ROOT}/{split}"
            x = (k % COLS) * PAIR_W
            y = (k // COLS) * TH
            sheet.paste(thumb(f"{base}/{split}_src_{idx:04d}.png"), (x, y))
            sheet.paste(thumb(f"{base}/{split}_dst_{idx:04d}.png"), (x + TH, y))
            d.rectangle([x + 3, y + 3, x + 120, y + 36], fill=(200, 0, 0))
            d.text((x + 10, y + 4), f"#{idx:04d}", fill=(255, 255, 255), font=FB)
            d.line([(x + PAIR_W - 2, y), (x + PAIR_W - 2, y + TH)], fill=(0, 220, 0), width=3)
        out = f"{OUT_DIR}/dense_{split}_{sh:02d}.png"
        sheet.save(out)
        print(out, flush=True)


build("train", 141)
build("test", 20)
print("DONE")
