# Qwen-Image-Edit-2511 × HQ-Edit LoRA 微调——最终报告

> 日期：2026-09-28
> 环境：4×RTX 3090 (24G) / CUDA 12.9 / PyTorch 2.13.0+cu129 / DiffSynth-Studio 2.1.7
> 项目目录：`/home/liufeng/projects/qwen-image-edit-sft`（源码）+ `/data/liufeng/qwen-image-edit-sft`（数据/模型/结果）

## 1. 全流程回顾

| 阶段 | 结果 |
|---|---|
| 数据准备 | HQ-Edit 子集 161 条（141 训练 + 20 测试），全量审计修复 1 条方向反转样本（#0000） |
| Baseline 推理 | 20/20，float8+CPU offload，平均 250s/张 |
| LoRA 训练 | 353/353 步 × 2 epochs，干净数据，2h51m，GPU 0/1 各 20.5GB@100% |
| LoRA 推理（A/B） | 20/20，3 卡分片并行，平均 266s/张 |
| 定量评估 | PSNR / SSIM / LPIPS 三指标 × gen-GT / gen-src 两组 |

## 2. 定量结果：LoRA vs Baseline（20 条测试集）

### 与目标图 GT 的接近程度（编辑保真度，核心指标）

| 指标 | Baseline | LoRA | 变化 |
|---|---|---|---|
| PSNR gen-GT ↑ | 10.14 | 10.30 | **+1.6%** |
| SSIM gen-GT ↑ | 0.2213 | 0.2313 | **+4.5%** |
| LPIPS gen-GT ↓ | 0.6507 | 0.6433 | **-1.1%** |

**逐样本胜负**：LoRA 在 **13/20** 样本上 LPIPS 优于 baseline。

**源图保留度**：LPIPS gen-src 0.4627 → 0.4456（-3.7%），即 LoRA 输出对源图结构保留更好——同时 GT 对齐度还提升了，说明学到的不是"少改"，而是"改得更准"。

### 逐样本 LPIPS gen-GT 明细（越低越好）

| idx | baseline | LoRA | 胜者 | | idx | baseline | LoRA | 胜者 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7137 | 0.6778 | LoRA | | 10 | 0.6589 | 0.6647 | baseline |
| 1 | 0.4602 | 0.4253 | LoRA | | 11 | 0.6236 | 0.6125 | LoRA |
| 2 | 0.7394 | 0.7270 | LoRA | | 12 | 0.6474 | 0.6332 | LoRA |
| 3 | 0.2958 | 0.2974 | baseline | | 13 | 0.7488 | 0.7501 | baseline |
| 4 | 0.7576 | 0.7553 | LoRA | | 14 | 0.7296 | 0.7091 | LoRA |
| 5 | 0.6897 | 0.6915 | baseline | | 15 | 0.6359 | 0.6665 | baseline |
| 6 | 0.6839 | 0.6818 | LoRA | | 16 | 0.6079 | 0.6108 | baseline |
| 7 | 0.6258 | 0.6004 | LoRA | | 17 | 0.6563 | 0.6446 | LoRA |
| 8 | 0.6835 | 0.6935 | baseline | | 18 | 0.6402 | 0.6274 | LoRA |
| 9 | 0.7523 | 0.7430 | LoRA | | 19 | 0.6631 | 0.6533 | LoRA |

## 3. 如何解读

- **方向一致的正向收益**：三项 GT 对齐指标全部改善 + 过半样本胜出 + 源图保留同步提高，改善是真实的，但**幅度温和**。
- **为什么幅度不大**：Qwen-Image-Edit-2511 基座本身已是 SOTA 级指令编辑模型，HQ-Edit（DALL-E 3 合成风格）对它属于"接近原生分布"的任务；141 条 × 2 epochs 的轻量 LoRA（rank 32）能带来的是精修而非质变。这个量级的提升对 20 样本的测试集来说方向可信，单看个别样本会受噪声影响。
- 对比官方论文在 HQ-Edit 上的全量微调动辄数千样本，本次实验定位是**跑通流程 + 验证链路正确**，这个目标已完全达成。

## 4. 产物位置

| 内容 | 路径 |
|---|---|
| **四联对比图（建议先看）** | `/data/liufeng/qwen-image-edit-sft/results/comparison_final/compare_0000~0019.png` |
| LoRA 权重 | `/data/liufeng/qwen-image-edit-sft/models/Qwen-Image-Edit-2511_lora_hqedit/epoch-1.safetensors`（epoch-0 也在） |
| 指标明细 | `results/lora/metrics.json`、`results/baseline/metrics.json` |
| 污染旧权重（可删） | `models/backup_polluted_20260924/`（约 900MB） |

### 关键脚本（均在 `/home/liufeng/projects/qwen-image-edit-sft/`）

| 脚本 | 用途 |
|---|---|
| `download_hqedit.py` | 下载 HQ-Edit 子集并转 DiffSynth 格式（含 `SWAP_INPUT_OUTPUT_ROWS={0}` 修复） |
| `train_lora_hqedit.sh` | LoRA 训练入口（2 进程 DDP + model_cpu_offload + expandable_segments） |
| `infer_testset.py` | 推理脚本（float8+CPU offload，支持分片/--lora） |
| `infer_parallel.sh` | 多卡并行推理启动器（`NUM_SHARDS=3`，已修复模型路径死锁问题） |
| `eval_metrics.py` | PSNR/SSIM/LPIPS 定量评估 |
| `make_comparison.py` | 生成 源A \| GT \| baseline \| LoRA 四联对比图 |

## 5. 训练配置备忘

```bash
# 核心参数（完整见 train_lora_hqedit.sh）
accelerate launch --num_processes 2 --gpu_ids 0,1 train.py \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/...,Qwen/Qwen-Image:text_encoder/...,Qwen/Qwen-Image:vae/..." \
  --enable_model_cpu_offload \
  --learning_rate 1e-4 --num_epochs 2 --dataset_repeat 5 \
  --lora_rank 32 --lora_target_modules "to_q,to_k,to_v,add_q_proj,..." \
  --use_gradient_checkpointing --zero_cond_t
```

关键环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（消除显存碎片）、`DIFFSYNTH_MODEL_BASE_PATH=/data/liufeng/models`。

## 6. 踩坑记录（复用时注意）

1. **HQ-Edit row 0 数据方向反转**：input_image/output_image 与 edit 指令相反，全量审计（肉眼+VLM 双交叉）161 条中仅此一条，已在下载脚本中修复；
2. **4 进程 DDP 会被 OOM Killer 杀**：每进程需完整权重副本（~54GB），4×54GB > 125GB 物理内存，用 2 进程；
3. **显存碎片浪费 7.77GB**：reserved-but-unallocated 导致首步 backward OOM，`expandable_segments:True` 解决；
4. **多进程推理必须设 `DIFFSYNTH_MODEL_BASE_PATH`**：否则回退 modelscope 在线下载，多进程抢文件锁死锁（3 小时 0 产出）；
5. **推理并行度上限 3**：每进程 CPU 常驻 ~31GB float8 权重，4 进程 124GB 逼近 125GB 上限；
6. **`--offload_models` 有框架 bug**：VAE 的 QwenImageCausalConv3d 从 disk reload 报 Unexpected key(s)，勿用；
7. **conda 克隆环境 shebang 陷阱**：入口脚本仍指向源环境 python，需批量修正。

## 7. 后续建议

1. **肉眼看图**：定量指标之外，建议翻一遍 `comparison_final/`，重点看 0008（车→摩托）、0017（缩小猫+放大植物）这类结构改动大的样本；
2. **想再压榨效果**：可以加 `--dataset_repeat`/epochs（当前 5×2），或提高样本量到全量 HQ-Edit；
3. **复用这套链路**：换自己的 A→B 数据时只需按 metadata.json 格式准备（`{"image": 目标B, "edit_image": 源A, "prompt": 编辑指令}`），训练/推理/评估脚本全部通用。
