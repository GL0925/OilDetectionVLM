#!/usr/bin/env python3
"""
测试Qwen3-VL图片描述能力
"""
import sys
from pathlib import Path
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

def main():
    # 模型路径
    model_path = "/home/amax/HDD1/gl_user/GL/mllm/qwen3-vl-8b-instruct"

    # 图片路径
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        # 默认用一张训练图片
        image_path = "/home/amax/HDD1/gl_user/GL/mllm/data/oil_datasets/test/image/os/155.png"

    print(f"模型: {model_path}")
    print(f"图片: {image_path}")

    if not Path(image_path).exists():
        print(f"图片不存在: {image_path}")
        return

    # 加载模型
    print("加载模型中...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # 构造对话
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "读取该图片，这个SAR图像中有没有溢油现象？大概在哪个地方？"}
            ]
        }
    ]

    # 处理输入
    print("生成描述中...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 生成
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # 打印结果
    print("\n" + "="*60)
    print("模型输出:")
    print("="*60)
    print(output_text[0])

if __name__ == "__main__":
    main()
