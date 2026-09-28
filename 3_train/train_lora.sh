#!/bin/bash
# Qwen-Image-Edit-2511 LoRA 训练脚本
# 基于 DiffSynth-Studio 官方示例，适配本地路径
# 项目目录: /home/liufeng/projects/qwen-image-edit-sft/
# 数据目录: /data/liufeng/qwen-image-edit-sft/ (大文件: 模型权重、数据集、checkpoint)

export DIFFSYNTH_MODEL_BASE_PATH=/data/liufeng/models
export CUDA_HOME=/home/liufeng/cuda129-nvcc
export TOKENIZERS_PARALLELISM=false

# 项目根目录
PROJECT_DIR=/home/liufeng/projects/qwen-image-edit-sft
# 数据根目录（大文件）
DATA_DIR=/data/liufeng/qwen-image-edit-sft

cd $PROJECT_DIR/DiffSynth-Studio

# 单卡 + CPU offload 模式（20B 模型在 3090 24G 上必须开启）
CUDA_VISIBLE_DEVICES=0 /home/liufeng/conda_envs/diffsynth/bin/python -m torch.distributed.run \
  --nproc_per_node=1 --nnodes=1 \
  examples/qwen_image/model_training/train.py \
  --dataset_base_path $DATA_DIR/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
  --dataset_metadata_path $DATA_DIR/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$DATA_DIR/models/Qwen-Image-Edit-2511_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 4 \
  --find_unused_parameters \
  --zero_cond_t \
  --enable_model_cpu_offload
