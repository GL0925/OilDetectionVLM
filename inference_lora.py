#!/usr/bin/env python3
"""
SAR溢油检测 推理评估脚本（LoRA微调后）

用法:
    python inference_lora.py \
        --data_root data/oil_datasets_split \
        --split test \
        --lora_path output/lora_weights \
        --epoch 2 \
        --output output/predictions_lora.json
"""

import os
import sys
from pathlib import Path

# 提前读取config，设置GPU环境变量（必须在import torch之前）
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from config import (MODEL_PATH, DATA_ROOT, LORA_OUTPUT_DIR,
                    PREDICTIONS_OUTPUT, EVAL_OUTPUT, SPLIT,
                    CUDA_VISIBLE_DEVICES, EPOCH)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import json
import argparse
from vlm_utils import parse_response, compute_metrics, print_metrics

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="SAR溢油检测 推理评估（LoRA）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--lora_path", default=str(LORA_OUTPUT_DIR))
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--split", default=SPLIT, choices=["train", "test"])
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="0=用最终权重，>0=指定epoch轮次（如2 → epoch-2）")
    parser.add_argument("--output", default=PREDICTIONS_OUTPUT)
    parser.add_argument("--eval_output", default=EVAL_OUTPUT)
    args = parser.parse_args()

    project_root = Path(__file__).parent

    # 根据epoch参数确定lora权重路径
    if args.epoch > 0:
        lora_path = project_root / args.lora_path / f"epoch-{args.epoch}"
        if not lora_path.exists():
            raise FileNotFoundError(f"epoch-{args.epoch} 权重不存在: {lora_path}")
        # 自动调整输出文件名，避免覆盖
        base, ext = Path(args.output).stem, Path(args.output).suffix
        args.output = str(Path(args.output).parent / f"{base}_epoch{args.epoch}{ext}")
        base_e, ext_e = Path(args.eval_output).stem, Path(args.eval_output).suffix
        args.eval_output = str(Path(args.eval_output).parent / f"{base_e}_epoch{args.epoch}{ext_e}")
    else:
        lora_path = project_root / args.lora_path

    # 加载基础模型
    model_path = Path(args.model_path)
    print(f"加载基础模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype="auto", device_map="auto"
    )

    # 加载LoRA权重
    print(f"加载LoRA权重: {lora_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()  # 合并LoRA权重
    print("模型加载完成\n")

    # 加载处理器
    processor = AutoProcessor.from_pretrained(str(model_path))

    # 收集图片
    data_root = project_root / args.data_root
    images = []
    for subdir in ["os", "no_os"]:
        img_dir = data_root / args.split / subdir / "image"
        if img_dir.exists():
            for img in sorted(img_dir.glob("*.png")):
                images.append(str(img))

    print(f"共 {len(images)} 张图片待推理\n")

    # 推理
    predictions = []
    prompt = "请分析这张SAR图像中的溢油情况，先简要分析再回答以下问题：\n1. 是否有溢油？\n2. 有几块溢油区域？\n3. 各区域大概在什么位置？\n\n请按以下格式回答，先在<analysis>中简要分析，再在<answer>中给出结论：\n<analysis>你的分析过程</analysis>\n<answer>\n1. 是否有溢油：有/无\n2. 区域数量：数字\n3. 位置：用九宫格方位描述（左上、中上、右上、左中、正中、右中、左下、中下、右下，多个位置用顿号分隔；若同一位置有多块，写成\"左上（2处）\"格式；若位置大于5个，写\"多处\"）\n</answer>"

    for i, image_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {Path(image_path).name} ...", end=" ")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        parsed = parse_response(response)
        parsed["image"] = image_path
        parsed["response"] = response
        predictions.append(parsed)

        status = "✓" if parsed["os"] is not None else "✗解析失败"
        print(status)

    # 保存预测结果
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n预测结果 → {output_path}")

    # 计算指标
    labels_path = data_root / f"{args.split}_labels.json"
    if labels_path.exists():
        labels = json.load(open(labels_path, encoding="utf-8"))
        metrics = compute_metrics(labels, predictions)
        print_metrics(metrics)
        eval_path = project_root / args.eval_output
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"评估结果 → {eval_path}")


if __name__ == "__main__":
    main()
