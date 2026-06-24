# SAR溢油检测 VLM微调项目

基于 Qwen3-VL-8B + LoRA 的SAR图像溢油检测微调模型。

[English](README.md) | 中文

## 环境依赖

```bash
torch>=2.0.0
modelscope
peft
transformers
matplotlib
Pillow
```

## 快速开始

### 1. 下载模型

```bash
# 使用modelscope下载qwen3-vl-8b-instruct到项目根目录
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./qwen3-vl-8b-instruct
```

### 2. 训练模型

```bash
python train.py
```

### 3. 推理评估

```bash
python inference_lora.py
```

## 训练结果

### 微调前后对比

| 任务 | 指标 | 微调前 | 微调后 (LoRA) | 提升 |
|------|------|--------|---------------|------|
| **溢油检测** | Accuracy | 0.6357 | 0.9457 | +0.3100 |
| | Precision | 0.7143 | 0.9889 | +0.2746 |
| | Recall | 0.8421 | 0.9368 | +0.0947 |
| | F1 | 0.7729 | 0.9622 | +0.1893 |
| **区域计数** | MAE | 2.0233 | 0.5426 | -1.4807 |
| | Exact Match | 0.2093 | 0.6124 | +0.4031 |
| **位置定位** | Hit Rate | 0.5895 | 0.8842 | +0.2947 |
| | Exact Match | 0.0632 | 0.4632 | +0.4000 |
| | Mean IoU | 0.2689 | 0.6860 | +0.4171 |

### 对比：Qwen-3.7-Max（原生）

| 任务 | 指标 | 数值 |
|------|------|------|
| **溢油检测** | Accuracy | 0.8992 |
| | Precision | 1.0000 |
| | Recall | 0.8632 |
| | F1 | 0.9266 |
| | TP/FP/TN/FN | 82/0/34/13 |
| **区域计数** | MAE | 0.8295 |
| | Exact Match | 0.5736 |
| **位置定位** | Hit Rate | 0.6105 |
| | Exact Match | 0.1368 |
| | Mean IoU | 0.3369 |

### VQA示例

**示例1：溢油检测（图片 1031.png）**

真实标签：
- 是否有溢油：有
- 区域数量：1
- 位置：左中、左下

模型回答：
<analysis>
对图像进行逐区域扫描，在左中、左下识别出1处暗色异常区域，面积适中。区域呈条状分布。根据暗色异常的分布和形态特征，符合溢油的SAR成像特性。判定为溢油。
</analysis>

- 是否有溢油：有
- 区域数量：1
- 位置：左中、左下


**示例2：无溢油（图片 13.png）**

真实标签：
- 是否有溢油：无
- 区域数量：0
- 位置：无

模型回答：
<analysis>
观察海面整体灰度分布，海面纹理均匀，未见暗色异常区域。判定为无溢油。
</analysis>

- 是否有溢油：无
- 区域数量：0
- 位置：无


## 项目结构

```
├── train.py              # 训练脚本
├── inference.py          # 基础模型推理
├── inference_lora.py     # LoRA微调后推理评估
├── config.py             # 配置文件
├── scripts/
│   ├── dataset.py        # 数据集加载
│   └── vlm_utils.py      # 工具函数
├── data/                 # 数据集目录
├── output/               # 输出目录
└── qwen3-vl-8b-instruct/ # 模型权重（git忽略）
```
