#!/usr/bin/env python
"""基线结果定量评估：PSNR / SSIM / LPIPS

三组对比:
- gen vs GT   : 编辑保真度（模型输出与期望结果的接近程度）
- gen vs src  : 源图保留度（输出相对源图的变化幅度，越小=保留越多）
- src vs GT   : 编辑幅度参考（该编辑指令本身的变化量，用于判断模型是否"敢改"）

运行环境: /data/liufeng/conda_envs/hch/bin/python（含 lpips/skimage/torch）
"""
import argparse
import json
import os

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SIZE = 512  # 统一评估分辨率


def load_img(path):
    img = Image.open(path).convert("RGB")
    return img.resize((SIZE, SIZE), Image.BICUBIC)


def to_numpy(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="/data/liufeng/qwen-image-edit-sft/results/baseline")
    parser.add_argument("--summary", type=str,
                        default="/data/liufeng/qwen-image-edit-sft/results/baseline/summary.json")
    parser.add_argument("--output", type=str,
                        default="/data/liufeng/qwen-image-edit-sft/results/baseline/metrics.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    lpips_fn = lpips.LPIPS(net="alex").to(args.device)

    rows = []
    for item in summary:
        i = item["index"]
        src = load_img(item["source"])
        gt = load_img(item["ground_truth"])
        gen_path = os.path.join(args.results_dir, f"test_{i:04d}.png")
        gen = load_img(gen_path)

        a_src, a_gt, a_gen = to_numpy(src), to_numpy(gt), to_numpy(gen)

        def np_psnr(a, b):
            return peak_signal_noise_ratio(a, b, data_range=1.0)

        def np_ssim(a, b):
            return structural_similarity(a, b, channel_axis=2, data_range=1.0)

        def t_lpips(a, b):
            ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(args.device) * 2 - 1
            tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).to(args.device) * 2 - 1
            with torch.no_grad():
                return float(lpips_fn(ta, tb).item())

        row = {
            "index": i,
            "prompt": item["prompt"],
            "psnr_gen_gt": float(np_psnr(a_gen, a_gt)),
            "ssim_gen_gt": float(np_ssim(a_gen, a_gt)),
            "lpips_gen_gt": t_lpips(a_gen, a_gt),
            "psnr_gen_src": float(np_psnr(a_gen, a_src)),
            "ssim_gen_src": float(np_ssim(a_gen, a_src)),
            "lpips_gen_src": t_lpips(a_gen, a_src),
            "lpips_src_gt": t_lpips(a_src, a_gt),
            "seconds": item.get("seconds"),
        }
        rows.append(row)
        print(f"[{i:04d}] PSNR(gen,GT)={row['psnr_gen_gt']:.2f} SSIM={row['ssim_gen_gt']:.4f} "
              f"LPIPS(gen,GT)={row['lpips_gen_gt']:.4f} | LPIPS(gen,src)={row['lpips_gen_src']:.4f} "
              f"LPIPS(src,GT)={row['lpips_src_gt']:.4f}")

    def agg(key):
        vals = [r[key] for r in rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}
    metrics = {
        "n": len(rows),
        "per_sample": rows,
        "aggregate": {k: agg(k) for k in
                      ["psnr_gen_gt", "ssim_gen_gt", "lpips_gen_gt",
                       "psnr_gen_src", "ssim_gen_src", "lpips_gen_src", "lpips_src_gt"]},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    a = metrics["aggregate"]
    print("\n===== 汇总 (n=%d) =====" % len(rows))
    print(f"编辑保真度  gen vs GT : PSNR {a['psnr_gen_gt']['mean']:.2f}±{a['psnr_gen_gt']['std']:.2f} dB | "
          f"SSIM {a['ssim_gen_gt']['mean']:.4f} | LPIPS {a['lpips_gen_gt']['mean']:.4f}")
    print(f"源图保留度  gen vs src: PSNR {a['psnr_gen_src']['mean']:.2f} | "
          f"SSIM {a['ssim_gen_src']['mean']:.4f} | LPIPS {a['lpips_gen_src']['mean']:.4f}")
    print(f"编辑幅度参考 src vs GT : LPIPS {a['lpips_src_gt']['mean']:.4f}")
    print(f"\n指标已保存 -> {args.output}")


if __name__ == "__main__":
    main()
