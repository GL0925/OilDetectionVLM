#!/usr/bin/env python3
"""
SAR溢油检测 推理评估脚本（原始模型）

用法:
    python inference.py \
        --data_root data/oil_datasets_split \
        --split test \
        --output output/predictions.json
"""

import os
import sys
from pathlib import Path

# 提前读取config，设置GPU环境变量（必须在import torch之前）
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from config import CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import json
import argparse
from vlm_utils import parse_response, compute_metrics, print_metrics

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="SAR溢油检测 推理评估")
    parser.add_argument("--model_path", default="qwen3-vl-8b-instruct")
    parser.add_argument("--data_root", default="data/oil_datasets_split")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", default="output/test_predictions.json")
    parser.add_argument("--eval_output", default="output/test_results.json")
    args = parser.parse_args()

    project_root = Path(__file__).parent

    # 加载模型
    model_path = project_root / args.model_path
    print(f"加载模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    print("模型加载完成\n")

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
