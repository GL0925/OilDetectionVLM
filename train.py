#!/usr/bin/env python3
"""
SAR溢油检测 VLM微调训练脚本

使用Qwen3-VL + LoRA进行微调
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch

# 添加scripts目录
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from config import *

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# 添加scripts目录
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from dataset import OilSpillDataset, collate_fn
from config import *


def setup_lora(model):
    """配置LoRA"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train():
    """训练主函数"""
    # 创建输出目录
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAR溢油检测 VLM微调训练")
    print("=" * 60)
    print(f"  模型: {MODEL_PATH}")
    print(f"  数据: {DATA_ROOT}")
    print(f"  输出: {LORA_OUTPUT_DIR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("=" * 60 + "\n")

    # 加载模型和处理器
    print("加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 配置LoRA
    print("\n配置LoRA...")
    model = setup_lora(model)

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = OilSpillDataset(DATA_ROOT, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练循环
    print("\n开始训练...")
    model.train()
    global_step = 0
    epoch_losses = []
    all_losses = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # 处理每条数据
            total_loss = 0.0

            for i in range(len(batch["image"])):
                image_path = batch["image"][i]
                conversations = batch["conversations"][i]

                # 构造完整对话（user + assistant）
                user_msg = conversations[0]["content"]
                assistant_msg = conversations[1]["content"]

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_msg.replace("<image>\n", "")}
                    ]},
                    {"role": "assistant", "content": assistant_msg}
                ]

                # 一次处理完整对话
                full_inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt"
                )
                full_inputs = full_inputs.to(model.device)

                # 通过user部分获取prompt长度
                user_only_messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_msg.replace("<image>\n", "")}
                    ]}
                ]
                user_inputs = processor.apply_chat_template(
                    user_only_messages, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt"
                )
                prompt_length = user_inputs["input_ids"].shape[1]

                # 创建labels，只计算assistant部分
                labels = full_inputs["input_ids"].clone()
                labels[:, :prompt_length] = -100  # 忽略prompt部分（batch维, seq维）

                # 前向传播（解包full_inputs，包含pixel_values等图像信息）
                outputs = model(**full_inputs, labels=labels)

                loss = outputs.loss / BATCH_SIZE
                total_loss += loss

                # 第一个batch打印详细信息
                if epoch == 0 and batch_idx == 0 and i == 0:
                    print("\n" + "=" * 60)
                    print("  第一个Batch调试信息")
                    print("=" * 60)
                    print(f"  图片路径: {image_path}")
                    print(f"  prompt_length: {prompt_length}")
                    print(f"  完整序列长度: {full_inputs['input_ids'].shape[1]}")
                    print(f"  labels非-100数量: {(labels != -100).sum().item()}")
                    print(f"  assistant长度: {full_inputs['input_ids'].shape[1] - prompt_length}")
                    print(f"  Loss: {outputs.loss.item():.4f}")
                    # 检查full_inputs原始token
                    full_ids = full_inputs["input_ids"][0].tolist()
                    print(f"  full_input_ids前20: {full_ids[:20]}")
                    print(f"  full_input_ids后20: {full_ids[-20:]}")
                    print(f"  full_input_ids在prompt_length处: {full_ids[prompt_length-3:prompt_length+3]}")
                    print(f"  labels前20: {labels[0].tolist()[:20]}")
                    print("=" * 60 + "\n")

            # 反向传播
            total_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
            total_loss.backward()

            # 梯度累积
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

            # 记录loss
            all_losses.append(total_loss.item() * GRADIENT_ACCUMULATION_STEPS)

            # 打印进度
            if (batch_idx + 1) % LOGGING_STEPS == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        # Epoch结束
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"\nEpoch {epoch+1}/{EPOCHS} 完成 | Average Loss: {avg_epoch_loss:.4f}\n")

    # 保存最终模型
    print("保存最终模型...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    processor.save_pretrained(LORA_OUTPUT_DIR)

    # 画loss曲线
    print("绘制loss曲线...")
    plt.figure(figsize=(12, 5))

    # 子图1: 每步loss
    plt.subplot(1, 2, 1)
    plt.plot(all_losses, alpha=0.3, label="Step Loss")
    # 平滑曲线
    window = min(50, len(all_losses))
    if window > 1:
        smoothed = [sum(all_losses[max(0, i-window):i+1]) / min(i+1, window)
                   for i in range(len(all_losses))]
        plt.plot(smoothed, label="Smoothed Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss (Per Step)")
    plt.legend()
    plt.grid(True)

    # 子图2: 每epoch平均loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), epoch_losses, marker="o", label="Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Per Epoch)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    print(f"Loss曲线已保存 → {OUTPUT_DIR / 'loss_curve.png'}")

    print("\n训练完成！")


if __name__ == "__main__":
    train()
