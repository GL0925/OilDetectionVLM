# SAR Oil Spill Detection - VLM Fine-tuning Project

Qwen3-VL-8B + LoRA based fine-tuning for SAR oil spill detection.

English | [中文](README_CN.md)

## Requirements

```bash
torch>=2.0.0
modelscope
peft
transformers
matplotlib
Pillow
```

## Quick Start

### 1. Download Model

```bash
# Download qwen3-vl-8b-instruct to project root using modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./qwen3-vl-8b-instruct
```

### 2. Training

```bash
python train.py
```

### 3. Inference & Evaluation

```bash
python inference_lora.py
```

## Training Results

### Before vs After Fine-tuning

| Task | Metric | Before | After (LoRA) | Improvement |
|------|--------|--------|--------------|-------------|
| **Oil Spill Detection** | Accuracy | 0.6357 | 0.9457 | +0.3100 |
| | Precision | 0.7143 | 0.9889 | +0.2746 |
| | Recall | 0.8421 | 0.9368 | +0.0947 |
| | F1 | 0.7729 | 0.9622 | +0.1893 |
| **Region Counting** | MAE | 2.0233 | 0.5426 | -1.4807 |
| | Exact Match | 0.2093 | 0.6124 | +0.4031 |
| **Location Detection** | Hit Rate | 0.5895 | 0.8842 | +0.2947 |
| | Exact Match | 0.0632 | 0.4632 | +0.4000 |
| | Mean IoU | 0.2689 | 0.6860 | +0.4171 |

### Comparison: Qwen-3.7-Max (Native)

| Task | Metric | Value |
|------|--------|-------|
| **Oil Spill Detection** | Accuracy | 0.8992 |
| | Precision | 1.0000 |
| | Recall | 0.8632 |
| | F1 | 0.9266 |
| | TP/FP/TN/FN | 82/0/34/13 |
| **Region Counting** | MAE | 0.8295 |
| | Exact Match | 0.5736 |
| **Location Detection** | Hit Rate | 0.6105 |
| | Exact Match | 0.1368 |
| | Mean IoU | 0.3369 |

### VQA Examples

**Example 1: Oil Spill (Image 1031.png)**

Ground Truth:
- Oil Spill: Yes
- Number of regions: 1
- Location: center-left, bottom-left

Model Prediction:
<analysis>
Scanning the image region by region, 1 dark anomaly was identified in the center-left and bottom-left areas, with moderate size. The regions appear elongated. Based on the distribution and morphological characteristics of the dark anomalies, they match the SAR imaging characteristics of oil spills. Determined to be oil spill.
</analysis>

- Oil Spill: Yes
- Number of regions: 1
- Location: center-left, bottom-left


**Example 2: No Oil Spill (Image 13.png)**

Ground Truth:
- Oil Spill: No
- Number of regions: 0
- Location: N/A

Model Prediction:
<analysis>
Observing the overall gray-level distribution of the sea surface, the texture is uniform with no visible dark anomaly regions. Determined to be no oil spill.
</analysis>

- Oil Spill: No
- Number of regions: 0
- Location: N/A


## Project Structure

```
├── train.py              # Training script
├── inference.py          # Base model inference
├── inference_lora.py     # LoRA fine-tuned inference & evaluation
├── config.py             # Configuration file
├── scripts/
│   ├── dataset.py        # Dataset loading
│   └── vlm_utils.py      # Utility functions
├── data/                 # Dataset directory
├── output/               # Output directory
└── qwen3-vl-8b-instruct/ # Model weights (git ignored)
```
