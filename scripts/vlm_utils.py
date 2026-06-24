"""
VLM溢油检测通用工具模块

包含:
- 答案解析器
- 评估指标计算
- 常量定义
"""

import re
from collections import defaultdict

# ═══════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════

MULTI_NUM_CODE = 10       # "多个"的编码
MULTI_LOCATION_CODE = 9   # "多处"的编码

# 中文数字映射
CHINESE_NUM_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "壹": 1,
    "二": 2, "两": 2, "贰": 2, "俩": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
    "十": 10, "拾": 10,
}

# 中文方位名 → 九宫格编号（支持简化名和旧格式兼容）
POSITION_MAP = {
    # 简化方位名（训练/推理统一格式）
    "左上": 0,
    "中上": 1,
    "右上": 2,
    "左中": 3,
    "正中": 4,
    "右中": 5,
    "左下": 6,
    "中下": 7,
    "右下": 8,
    # 旧格式兼容（未微调模型可能输出）
    "左上角": 0,
    "上方中部": 1, "上中": 1,
    "右上角": 2,
    "左侧中部": 3,
    "正中央": 4, "中央": 4,
    "右侧中部": 5,
    "左下角": 6,
    "下方中部": 7, "下中": 7,
    "右下角": 8,
}


# ═══════════════════════════════════════════
#  答案解析器
# ═══════════════════════════════════════════

def _parse_number(text):
    """
    解析数字文本，支持阿拉伯数字和中文数字。
    例: "3" → 3, "三" → 3, "3处" → 3, "三个" → 3
    """
    text = text.strip()

    # 纯数字
    if text.isdigit():
        return int(text)

    # 中文数字
    if text in CHINESE_NUM_MAP:
        return CHINESE_NUM_MAP[text]

    # "X个"、"X块"、"X处"、"X条" 格式（阿拉伯数字）
    match = re.match(r'^(\d+)[个块处条]', text)
    if match:
        return int(match.group(1))

    # "X个"、"X块"、"X处"、"X条" 格式（中文数字）
    match = re.match(r'^([一二两三四五六七八九十百千万亿零〇壹贰叁肆伍陆柒捌玖拾佰仟]+)[个块处条]', text)
    if match:
        num_str = match.group(1)
        return CHINESE_NUM_MAP.get(num_str, None)

    return None


def parse_response(response_text):
    """
    从模型生成的文本中解析出结构化标签。

    Args:
        response_text: 模型生成的完整回复（含 <analysis>...<answer>...）

    Returns:
        dict: {"os": bool, "num": int, "location": list[int]}
              解析失败时对应字段为 None
    """
    result = {"os": None, "num": None, "location": None}

    # 提取 <answer> 部分（如果有的话）
    answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    text = answer_match.group(1).strip() if answer_match else response_text.strip()

    # ---- 解析 "是否有溢油" ----
    oil_match = re.search(r"是否有溢油[：:]\s*(有|无)", text)
    if oil_match:
        result["os"] = (oil_match.group(1) == "有")

    # ---- 解析 "区域数量" ----
    num_match = re.search(r"区域数量[：:]\s*(\S+)", text)
    if num_match:
        raw = num_match.group(1).strip()
        if "多" in raw:
            result["num"] = MULTI_NUM_CODE
        else:
            result["num"] = _parse_number(raw)

    # ---- 解析 "位置" ----
    loc_match = re.search(r"位置[：:]\s*(.+?)(?:\n|$)", text)
    if loc_match:
        raw_loc = loc_match.group(1).strip()
        if "多处" in raw_loc:
            if result["os"] is False:
                result["location"] = []
            else:
                result["location"] = [MULTI_LOCATION_CODE]
        elif "无溢油" in raw_loc:
            result["location"] = []
        else:
            # 尝试直接提取数字编号（新格式）
            locations = _parse_positions_numeric(raw_loc)
            if locations:
                result["location"] = locations
            else:
                # 回退到旧的自然语言解析
                locations = _parse_positions_natural(raw_loc)
                result["location"] = locations

    return result


def _parse_positions_numeric(text):
    """
    从位置描述中直接提取九宫格数字编号。
    例: "0,4,6" → [0, 4, 6]
        "位置：0, 4, 6" → [0, 4, 6]
    
    Returns:
        list[int] or None: 如果找到数字编号则返回，否则返回 None
    """
    # 匹配所有独立的数字（0-8）
    matches = re.findall(r'\b([0-8])\b', text)
    if matches:
        return sorted(set(int(m) for m in matches))
    return None


def _parse_positions_natural(text):
    """
    从自然语言位置描述中解析九宫格编号列表（旧格式兼容）。
    例: "左上（2处）、正中、右中，呈块状分布" → [0, 4, 5]
        "左上角（2处）、正中央、右侧中部" → [0, 4, 5]
    """
    locations = []

    # 移除形状描述后缀（逗号之后的内容）
    text = re.split(r"[，,]", text)[0] if "，" in text or "," in text else text

    # 匹配所有方位名（可能带"（N处）"或"(N处)"后缀，中英文括号都兼容）
    pattern = r"(" + "|".join(re.escape(k) for k in sorted(POSITION_MAP.keys(), key=len, reverse=True)) + r")(?:[（(]\d+处[）)])?"
    matches = re.findall(pattern, text)

    for pos_name in matches:
        if pos_name in POSITION_MAP:
            idx = POSITION_MAP[pos_name]
            if idx not in locations:
                locations.append(idx)

    return sorted(locations)


# ═══════════════════════════════════════════
#  评估指标
# ═══════════════════════════════════════════

def compute_metrics(labels, predictions):
    """
    计算三个子任务的评估指标。

    Args:
        labels: list of ground truth label dicts
        predictions: list of parsed prediction dicts

    Returns:
        dict of metrics
    """
    metrics = {}

    # 对齐数据（按 image path 匹配）
    label_map = {l["image"]: l for l in labels}
    paired = []
    for p in predictions:
        if p["image"] in label_map:
            paired.append((label_map[p["image"]], p))

    n = len(paired)
    if n == 0:
        return {"error": "no matched samples"}
    metrics["total_samples"] = n

    # ---- 任务1: 溢油检测 (os) ----
    tp = fp = tn = fn = 0
    for gt, pred in paired:
        if gt["os"] and pred["os"]:
            tp += 1
        elif gt["os"] and not pred["os"]:
            fn += 1
        elif not gt["os"] and pred["os"]:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics["detection"] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    # ---- 任务2: 区域计数 (num) ----
    maes = []
    num_exact = 0
    for gt, pred in paired:
        if gt["num"] is not None and pred["num"] is not None:
            maes.append(abs(gt["num"] - pred["num"]))
            if gt["num"] == pred["num"]:
                num_exact += 1

    if maes:
        metrics["counting"] = {
            "mae": round(sum(maes) / len(maes), 4),
            "exact_match_rate": round(num_exact / len(maes), 4),
            "evaluated": len(maes),
        }

    # ---- 任务3: 位置定位 (location) ----
    loc_hits = 0      # 至少命中一个格子
    loc_exact = 0     # 完全匹配
    loc_total = 0
    loc_iou_sum = 0.0

    for gt, pred in paired:
        gt_loc = set(gt["location"])
        pred_loc = set(pred["location"])

        # 跳过无溢油样本
        if not gt["os"]:
            continue
        loc_total += 1

        # 处理"多处"特殊编码
        if MULTI_LOCATION_CODE in gt_loc and MULTI_LOCATION_CODE in pred_loc:
            loc_hits += 1
            loc_exact += 1
            loc_iou_sum += 1.0
            continue
        elif MULTI_LOCATION_CODE in gt_loc or MULTI_LOCATION_CODE in pred_loc:
            # 一方说多处另一方没说 → 不匹配
            continue

        # 正常格子匹配
        if gt_loc and pred_loc:
            intersection = gt_loc & pred_loc
            union = gt_loc | pred_loc
            if intersection:
                loc_hits += 1
            if gt_loc == pred_loc:
                loc_exact += 1
            loc_iou_sum += len(intersection) / len(union)

    if loc_total > 0:
        metrics["localization"] = {
            "hit_rate": round(loc_hits / loc_total, 4),
            "exact_match_rate": round(loc_exact / loc_total, 4),
            "mean_iou": round(loc_iou_sum / loc_total, 4),
            "evaluated": loc_total,
        }

    return metrics


def print_metrics(metrics):
    """打印评估结果"""
    print("\n" + "=" * 50)
    print("  SAR 溢油检测 VLM 评估结果")
    print("=" * 50)
    print(f"  样本数: {metrics.get('total_samples', 0)}")

    if "detection" in metrics:
        d = metrics["detection"]
        print(f"\n  [任务1] 溢油检测")
        print(f"    Accuracy:  {d['accuracy']:.4f}")
        print(f"    Precision: {d['precision']:.4f}")
        print(f"    Recall:    {d['recall']:.4f}")
        print(f"    F1:        {d['f1']:.4f}")
        print(f"    TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']}")

    if "counting" in metrics:
        c = metrics["counting"]
        print(f"\n  [任务2] 区域计数")
        print(f"    MAE:            {c['mae']:.4f}")
        print(f"    Exact Match:    {c['exact_match_rate']:.4f}")
        print(f"    Evaluated:      {c['evaluated']}")

    if "localization" in metrics:
        l = metrics["localization"]
        print(f"\n  [任务3] 位置定位")
        print(f"    Hit Rate:       {l['hit_rate']:.4f}")
        print(f"    Exact Match:    {l['exact_match_rate']:.4f}")
        print(f"    Mean IoU:       {l['mean_iou']:.4f}")
        print(f"    Evaluated:      {l['evaluated']}")

    print()
