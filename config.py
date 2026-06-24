"""
SAR溢油检测 VLM微调配置
"""

from pathlib import Path

# ═══════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data" / "oil_datasets_split"
MODEL_PATH = PROJECT_ROOT / "qwen3-vl-8b-instruct"
OUTPUT_DIR = PROJECT_ROOT / "output"
LORA_OUTPUT_DIR = OUTPUT_DIR / "lora_weights"

# GPU配置
CUDA_VISIBLE_DEVICES = "4,5"  

# ═══════════════════════════════════════════
#  LoRA配置
# ═══════════════════════════════════════════
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",   # LLM Attention
                       "gate_proj", "up_proj", "down_proj",       # LLM MLP
                       "attn.qkv", "attn.proj",                   # ViT Attention
                       "linear_fc1", "linear_fc2"]                # ViT MLP

# ═══════════════════════════════════════════
#  训练配置
# ═══════════════════════════════════════════
EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

# ═══════════════════════════════════════════
#  推理配置
# ═══════════════════════════════════════════
SPLIT = "test"
EPOCH = 0  # 0=用最终权重，>0=用对应epoch权重（如2 → epoch-2）
PREDICTIONS_OUTPUT = str(OUTPUT_DIR / "predictions_lora.json")
EVAL_OUTPUT = str(OUTPUT_DIR / "predictions_result.json")

# ═══════════════════════════════════════════
#  其他配置
# ═══════════════════════════════════════════
SEED = 42
LOGGING_STEPS = 10
SAVE_STEPS = 50
FP16 = False
BF16 = True
