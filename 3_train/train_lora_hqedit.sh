#!/bin/bash
# Qwen-Image-Edit-2511 LoRA 训练脚本（HQ-Edit 数据集）
# 训练数据: 141 条 HQ-Edit 图像编辑对
# 测试数据: 20 条独立测试集（baseline 已跑完）
#
# 总步数 = 141 条 × repeat 5 × 2 epochs = 1410 步
# fp8 量化 DiT（40GB→20GB）使权重可常驻 GPU，避免 CPU offload 的 PCIe 搬运瓶颈

PYTHON=/home/liufeng/conda_envs/diffsynth/bin/python
export PATH=/home/liufeng/conda_envs/diffsynth/bin:$PATH
export DIFFSYNTH_MODEL_BASE_PATH=/data/liufeng/models
export CUDA_HOME=/home/liufeng/cuda129-nvcc
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
# expandable_segments 消除显存碎片化（否则 7.7GB reserved-unallocated 白白浪费导致 OOM）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/liufeng/projects/qwen-image-edit-sft/DiffSynth-Studio

# shebang 已修复为指向本环境 python，直接调用入口脚本
# 2 进程 DDP：4 进程会因每进程权重副本(~54GB)超出物理内存(125GB)被 OOM Killer 杀掉
/home/liufeng/conda_envs/diffsynth/bin/accelerate launch --num_processes 2 --gpu_ids 0,1 examples/qwen_image/model_training/train.py \
  --dataset_base_path /data/liufeng/qwen-image-edit-sft/data/hqedit/train \
  --dataset_metadata_path /data/liufeng/qwen-image-edit-sft/data/hqedit/train/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 5 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --enable_model_cpu_offload \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/data/liufeng/qwen-image-edit-sft/models/Qwen-Image-Edit-2511_lora_hqedit" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --zero_cond_t
