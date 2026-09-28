#!/usr/bin/env python
"""Qwen-Image-Edit-2511 测试集推理脚本

用法:
  baseline:  python infer_testset.py --output_dir /data/liufeng/qwen-image-edit-sft/results/baseline
  LoRA A/B:  python infer_testset.py --output_dir /data/liufeng/qwen-image-edit-sft/results/lora \
                --lora /data/liufeng/qwen-image-edit-sft/models/Qwen-Image-Edit-2511_lora_hqedit/epoch-1.safetensors

基于 DiffSynth-Studio 官方 low_vram 示例（float8 + CPU offload，单卡 24G 可跑）
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str,
                        default="/data/liufeng/qwen-image-edit-sft/data/hqedit/test/metadata.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None,
                        help="可选：LoRA 权重路径（不传则跑 baseline）")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0,
                        help="分片编号：多卡数据并行时每张卡处理一部分（与 --num_shards 配合）")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="总分片数（如 4 张卡各跑一个进程：--num_shards 4 --shard 0..3）")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # low-VRAM 配置：权重以 float8 常驻内存，逐块上 GPU 计算
    # （内存 114G 充裕；官方示例 offload 到 disk，这里用 cpu 更快）
    vram_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    print("加载模型（float8 + CPU offload）...")
    t0 = time.time()
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511",
                        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Qwen/Qwen-Image",
                        origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
            ModelConfig(model_id="Qwen/Qwen-Image",
                        origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    print(f"模型加载完成，耗时 {time.time()-t0:.0f}s")

    if args.lora:
        print(f"加载 LoRA: {args.lora}")
        pipe.load_lora(pipe.dit, args.lora)
        print("LoRA 加载完成")

    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # 数据并行分片：第 i 个分片处理 (全局索引, 条目) 对，输出文件名用全局索引避免冲突
    indexed_items = list(enumerate(metadata))
    if args.num_shards > 1:
        indexed_items = indexed_items[args.shard::args.num_shards]
        print(f"分片 {args.shard}/{args.num_shards}：本进程处理 {len(indexed_items)} 条\n")
    else:
        print(f"测试集 {len(indexed_items)} 条\n")

    results_summary = []
    for i, item in indexed_items:
        src_path = item["edit_image"]
        gt_path = item["image"]
        prompt = item["prompt"]

        edit_image = [Image.open(src_path).convert("RGB")]
        w, h = edit_image[0].size

        t1 = time.time()
        image = pipe(
            prompt,
            edit_image=edit_image,
            seed=args.seed,           # 固定 seed，保证 baseline / LoRA 可比
            num_inference_steps=args.num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=True,
            zero_cond_t=True,         # Qwen-Image-Edit-2511 专用参数
        )
        out_path = os.path.join(args.output_dir, f"test_{i:04d}.png")
        image.save(out_path)

        dt = time.time() - t1
        results_summary.append({
            "index": i, "source": src_path, "ground_truth": gt_path,
            "generated": out_path, "prompt": prompt, "seconds": round(dt, 1),
        })
        print(f"[index={i}] {dt:.0f}s -> {out_path}")

    # 多分片时各写各的 summary，避免互相覆盖；单进程保持 summary.json 兼容对比脚本
    if args.num_shards > 1:
        summary_path = os.path.join(args.output_dir, f"summary_shard{args.shard}.json")
    else:
        summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成。结果摘要 -> {summary_path}")


if __name__ == "__main__":
    main()
