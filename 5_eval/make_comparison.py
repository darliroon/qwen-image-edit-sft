#!/usr/bin/env python
"""生成 A/B 对比图：源图A | 目标图B(GT) | baseline输出 | LoRA输出 四联拼图

用法:
  python make_comparison.py \
    --baseline_dir /data/liufeng/qwen-image-edit-sft/results/baseline \
    --lora_dir /data/liufeng/qwen-image-edit-sft/results/lora \
    --output_dir /data/liufeng/qwen-image-edit-sft/results/comparison
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="可选：LoRA 结果目录（不传则只对比 源图|GT|baseline）")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--thumb_size", type=int, default=512, help="每格缩略图边长")
    return parser.parse_args()


def load_font(size=18):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.baseline_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)

    font = load_font(20)
    labels = ["Source (A)", "Ground Truth (B)", "Baseline"]
    if args.lora_dir:
        labels.append("LoRA")

    for item in summary:
        i = item["index"]
        cols = [Image.open(item["source"]).convert("RGB"),
                Image.open(item["ground_truth"]).convert("RGB"),
                Image.open(item["generated"]).convert("RGB")]
        if args.lora_dir:
            lora_path = os.path.join(args.lora_dir, f"test_{i:04d}.png")
            if os.path.exists(lora_path):
                cols.append(Image.open(lora_path).convert("RGB"))
            else:
                print(f"警告：缺少 LoRA 结果 {lora_path}，跳过该列")

        n = len(cols)
        ts = args.thumb_size
        header_h = 30
        canvas = Image.new("RGB", (ts * n, ts + header_h), "white")
        draw = ImageDraw.Draw(canvas)

        for c, (img, label) in enumerate(zip(cols, labels)):
            # 等比缩放裁剪到 ts×ts
            img.thumbnail((ts, ts))
            canvas.paste(img, (c * ts + (ts - img.width) // 2, header_h))
            draw.text((c * ts + 8, 5), label, fill="black", font=font)

        out_path = os.path.join(args.output_dir, f"compare_{i:04d}.png")
        canvas.save(out_path)

    print(f"已生成 {len(summary)} 张对比图 -> {args.output_dir}")
    print(f"prompt 清单：")
    for item in summary:
        print(f"  [{item['index']:04d}] {item['prompt'][:80]}")


if __name__ == "__main__":
    main()
