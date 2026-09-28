# Qwen-Image-Edit-2511 LoRA 微调项目（HQ-Edit）

在 4×RTX 3090 (24G) 上对 Qwen-Image-Edit-2511（20B 图像编辑模型）做 LoRA 微调的完整流水线。
数据集为 HQ-Edit 子集（指令式图像编辑对），目标：让模型学习"按指令大胆、完整地编辑"。

**最终结果**（详见 [docs/REPORT.md](docs/REPORT.md)）：LoRA 在 13/20 测试样本上 LPIPS 优于基线，编辑保真度与源图保留度同时提升。

```
全流程：  下载HQ-Edit ──→ 方向审计 ──→ LoRA训练 ──→ 测试集推理 ──→ 定量评估
对应目录：   1_data/       2_audit/      3_train/      4_infer/       5_eval/
```

---

## 一、迁移清单（换机需要带走什么）

| 内容 | 本机路径 | 大小 | 迁移方式 |
|---|---|---|---|
| 项目代码（本目录） | `/home/liufeng/projects/qwen-image-edit-sft/` | ~40M | 打包拷贝（含 DiffSynth-Studio/） |
| 基座模型 | `/data/liufeng/models/` | ~86G | 拷贝 或 重新下载（见第三节） |
| 数据集+LoRA权重+结果 | `/data/liufeng/qwen-image-edit-sft/` | ~5G | 打包拷贝（或按第四节重建） |
| Python 环境依赖清单 | `envs/requirements-*.txt`（本目录内） | - | 新机器按第二节重建 |

> 依赖清单是本机三个环境的完整 `pip freeze` 快照（2026-09-28），版本已验证可跑通全流程。

## 二、硬件要求（已验证的最低配置）

| 项目 | 要求 | 说明 |
|---|---|---|
| GPU | 4× RTX 3090 24GB | 训练用 2 卡（各 ~20.5GB 显存）；推理最多 3 卡分片 |
| 显卡驱动 | ≥ 535.274.02 | 本机实测版本；搭配 torch 2.13.0+cu129 |
| 内存 | ≥ 125GB（重要！） | 推理时每个分片进程在 CPU 常驻 ~31GB float8 权重，3 分片=93GB |
| 磁盘 | ≥ 100GB 空闲 | 模型 86G + 数据/产物 ~10G |
| CUDA 运行库 | 不需要单独装 | torch cu129 wheel 自带，只需驱动版本够新 |

## 三、环境搭建（新机器从零开始）

三个环境相互独立，作用不同。**只有环境①是必需的**，②③按需建。

### 环境① diffsynth —— 训练 + 推理（必需）

```bash
# 1. 创建环境（conda 或 micromamba 均可）
conda create -n diffsynth python=3.11 -y
conda activate diffsynth

# 2. 安装依赖（完整版本清单见 envs/requirements-diffsynth.txt）
#    cu129 索引必须加，否则 torch 装成 CPU 版或 cu130 版（驱动不兼容）
pip install -r envs/requirements-diffsynth.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129

# 3. editable 安装 DiffSynth-Studio（必须在本目录下执行！）
pip install -e DiffSynth-Studio

# 4. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期输出: 2.13.0+cu129 True
python -c "import diffsynth; print(diffsynth.__file__)"
# 预期输出: .../qwen-image-edit-sft/DiffSynth-Studio/diffsynth/__init__.py
```

关键版本（为什么是这些版本——已验证的组合，勿随意升级）：

| 包 | 版本 | 备注 |
|---|---|---|
| python | 3.11.16 | |
| torch / torchvision | 2.13.0+cu129 / 0.28.0+cu129 | cu130 与驱动 535 不兼容，勿装 |
| transformers | 5.8.0 | |
| peft | 0.18.1 | LoRA 加载 |
| accelerate | 1.11.0 | 训练启动器 |
| diffsynth | 2.1.7 (editable) | 本仓库 DiffSynth-Studio/，git 源: gitcode.com/modelscope/DiffSynth-Studio |

### 环境② qwen25vl —— 数据方向审计（可选，数据已审计过可不建）

```bash
conda create -n qwen25vl python=3.11 -y
conda activate qwen25vl
pip install -r envs/requirements-qwen25vl.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129
# 关键包: transformers 5.8.0, qwen-vl-utils 0.0.14 (Qwen2.5-VL 推理用)
```

### 环境③ hch —— 评估指标计算（可选，只算 PSNR/SSIM/LPIPS 时用）

```bash
conda create -n hch python=3.10 -y
conda activate hch
pip install -r envs/requirements-hch.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
# 关键包: lpips 0.1.4, scikit-image 0.26.0（纯 CPU 也能跑，torch 版本不敏感）
```

## 四、模型准备（~86G，二选一）

### 方式A：直接拷贝（推荐，快）

把本机 `/data/liufeng/models/` 下这四个目录整个拷到新机器同位置：

```
/data/liufeng/models/
├── Qwen/                        ← 关键！模型ID带组织名前缀 "Qwen/xxx"，代码按此查找
│   ├── Qwen-Image-Edit-2511     (符号链接或目录均可)
│   ├── Qwen-Image
│   └── Qwen-Image-Edit
├── Qwen-Image-Edit-2511/        # 54G: transformer/ 下 5 个 safetensors 分片 + config.json + index.json
├── Qwen-Image/                  # 16G: text_encoder/ + tokenizer/ + vae/
├── Qwen-Image-Edit/             # processor/ 配置 (很小但必需)
└── Qwen2.5-VL-7B-Instruct/      # 16G: 仅审计用，不跑审计可不拷
```

### 方式B：从 HuggingFace 重新下载（国内走镜像）

```bash
pip install -U "huggingface_hub[cli]"

# 国内镜像必须设这两个变量！
#   HF_ENDPOINT            → 走 hf-mirror 镜像
#   HF_HUB_DISABLE_XET=1   → 不设会 401 Unauthorized（Xet 协议 CDN 不通，踩过的坑）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 注意 local-dir 用 models/Qwen/<名字> 结构（对应模型ID "Qwen/Qwen-Image-Edit-2511"）
huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
    --local-dir /data/liufeng/models/Qwen/Qwen-Image-Edit-2511

huggingface-cli download Qwen/Qwen-Image \
    --include "text_encoder/*" "tokenizer/*" "vae/*" \
    --local-dir /data/liufeng/models/Qwen/Qwen-Image

huggingface-cli download Qwen/Qwen-Image-Edit \
    --include "processor/*" \
    --local-dir /data/liufeng/models/Qwen/Qwen-Image-Edit
```

### 模型验证

```bash
# 目录结构自检（三样齐全才算完整）
ls /data/liufeng/models/Qwen/Qwen-Image-Edit-2511/transformer/ | wc -l   # ≥7 (5分片+config+index)
ls /data/liufeng/models/Qwen/Qwen-Image/{text_encoder,vae,tokenizer} -d
ls /data/liufeng/models/Qwen/Qwen-Image-Edit/processor/
```

## 五、数据集准备（两种方式）

数据集位于 `/data/liufeng/qwen-image-edit-sft/data/hqedit/`：**train 141 对 + test 20 对**，
每对 = 源图(src) + 目标图(dst) + 编辑指令，另有 `metadata.json`（DiffSynth 格式）。

- **方式A（推荐）**：直接拷贝整个 `/data/liufeng/qwen-image-edit-sft/` 目录（含数据、LoRA 权重、结果，共 ~5G）
- **方式B（重建）**：`python 1_data/download_hqedit.py`（从 HF 下载 HQ-Edit 子集并转换格式，需镜像环境变量同上）

## 六、跑通全流程（5 个阶段）

> 所有训练/推理命令前必须设：
> ```bash
> export DIFFSYNTH_MODEL_BASE_PATH=/data/liufeng/models   # 模型查找根路径
> export HF_HUB_OFFLINE=1                                  # 禁止在线下载（多进程并发下载会死锁！）
> ```
> 建议写进 `~/.bashrc`。用环境①（conda activate diffsynth）。

### 阶段1 数据（已完成，跳过）
`1_data/download_hqedit.py` — 下载 HQ-Edit 子集并转成 DiffSynth 的 metadata.json 格式。

### 阶段2 方向审计（已完成，跳过；新数据集时必做）

**为什么审计**：HQ-Edit 有少量样本 A/B 方向反转（源图与目标图放反）。曾因 9 条 REVERSED 样本导致一轮污染训练（该 LoRA 已备份在 models/backup_polluted_20260924/，勿用）。

```bash
# 双卡并行审计（用环境② qwen25vl）
CUDA_VISIBLE_DEVICES=2 python 2_audit/audit_direction.py --shard 0 --num_shards 2 &
CUDA_VISIBLE_DEVICES=3 python 2_audit/audit_direction.py --shard 1 --num_shards 2
# 结果: 2_audit/outputs/audit_all.jsonl，每条含 FORWARD/REVERSED 判定
# 人工抽检: 2_audit/make_sheets.py 生成 src/dst 对比图 -> 2_audit/sheets/
```

### 阶段3 LoRA 训练（2 卡，~3 小时）

```bash
bash 3_train/train_lora_hqedit.sh
# 要点（脚本内已配置）:
#   - accelerate launch --num_processes 2 --gpu_ids 0,1
#   - --enable_model_cpu_offload  (20B 模型 24G 卡必须开，DDP 会 OOM)
#   - --zero_cond_t               (Qwen-Image-Edit-2511 必需参数)
#   - LoRA rank 32, 2 epochs, dataset_repeat 5
# 产出: /data/liufeng/qwen-image-edit-sft/models/Qwen-Image-Edit-2511_lora_hqedit/epoch-{0,1}.safetensors
#      (epoch-0 = 第1轮, epoch-1 = 第2轮; 实验中 epoch-1 更优, 详见 docs/REPORT.md)
# 日志: /data/liufeng/qwen-image-edit-sft/train_hqedit.log
```

### 阶段4 测试集推理（基线 + LoRA 各跑一次，同 seed=42 保证可比）

```bash
# 基线（不带 LoRA）
NUM_SHARDS=3 bash 4_infer/infer_parallel.sh /data/liufeng/qwen-image-edit-sft/results/baseline

# LoRA 版（指定权重路径）
NUM_SHARDS=3 bash 4_infer/infer_parallel.sh /data/liufeng/qwen-image-edit-sft/results/lora \
    /data/liufeng/qwen-image-edit-sft/models/Qwen-Image-Edit-2511_lora_hqedit/epoch-1.safetensors

# 说明: 每分片一个 GPU 进程，20 张图约 30 分钟
#       NUM_SHARDS 上限=3（内存限制，见第二节）；单卡跑用 NUM_SHARDS=1
```

### 阶段5 定量评估（用环境③ hch）

```bash
R=/data/liufeng/qwen-image-edit-sft/results   # 结果根目录

# 基线指标（PSNR/SSIM/LPIPS，512×512 统一口径）
python 5_eval/eval_metrics.py \
    --results_dir $R/baseline \
    --summary $R/baseline/summary.json \
    --output $R/baseline/metrics.json

# LoRA 版（同口径对比，三个参数都要改）
python 5_eval/eval_metrics.py \
    --results_dir $R/lora \
    --summary $R/lora/summary.json \
    --output $R/lora/metrics.json

# 生成 src|GT|输出 三联对比图（人工核查用）
python 5_eval/make_comparison.py
```

## 七、常见坑（全部实际踩过，迁移后遇到先查这里）

| 症状 | 原因与解法 |
|---|---|
| pip 装 torch 后 `torch.cuda.is_available()=False` 或驱动报错 | 装成了 cu130 版。必须 `--extra-index-url https://download.pytorch.org/whl/cu129` |
| huggingface 下载 401 Unauthorized | 未设 `HF_HUB_DISABLE_XET=1`（Xet CDN 不通） |
| 推理卡死不动 | 忘设 `DIFFSYNTH_MODEL_BASE_PATH`，多进程并发走在线下载 → 文件锁死锁。设好路径 + `HF_HUB_OFFLINE=1` |
| 训练 OOM | 20B 模型没用 cpu_offload；确认 train_lora_hqedit.sh 的 `--enable_model_cpu_offload` |
| 推理分片进程被 OOM-kill | 分片数超 3，内存不够（每分片常驻 31GB） |
| LoRA 效果极差/方向错乱 | 数据方向未审计。新数据必须先跑阶段2 |
| 显存碎片告警 | 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（不影响数值） |

## 八、迁移后自检清单（按顺序执行，全过=迁移成功）

```bash
# 1. 环境
conda activate diffsynth && python -c "import torch; assert torch.cuda.is_available(); print('torch OK')"
python -c "import diffsynth; print('diffsynth OK')"

# 2. 模型路径
export DIFFSYNTH_MODEL_BASE_PATH=/data/liufeng/models
ls $DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen-Image-Edit-2511/transformer/*.safetensors | wc -l   # 5

# 3. 数据
ls /data/liufeng/qwen-image-edit-sft/data/hqedit/train/metadata.json

# 4. 单张试推理（~5分钟，出图即全通）
cd 4_infer && CUDA_VISIBLE_DEVICES=0 python infer_testset.py \
    --output_dir /tmp/migration_check --num_shards 1 --shard 0
```

## 九、目录结构速查

```
qwen-image-edit-sft/
├── README.md                  ← 本文件
├── docs/REPORT.md             ← 最终实验报告（结论数据）
├── envs/                      ← 三个环境的 pip freeze 快照（迁移用）
├── 1_data/                    ← 阶段1: 数据下载与格式转换
├── 2_audit/                   ← 阶段2: 方向审计 (脚本 + outputs/ + sheets/)
├── 3_train/                   ← 阶段3: LoRA 训练脚本
├── 4_infer/                   ← 阶段4: 测试集推理 (单进程 + 多卡分片)
├── 5_eval/                    ← 阶段5: PSNR/SSIM/LPIPS 评估与对比图
└── DiffSynth-Studio/          ← DiffSynth 2.1.7 源码（editable install 锚定，勿移动）
```

`/data` 侧布局：`/data/liufeng/models/`（基座模型）+ `/data/liufeng/qwen-image-edit-sft/`（数据集 data/、LoRA 权重 models/、推理结果 results/）。
