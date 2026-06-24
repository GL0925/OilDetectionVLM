"""
SAR溢油检测数据集类

加载train.json/test.json（VQA格式），用于训练。
train_labels.json/test_labels.json仅用于评估，不参与训练。
"""

import json
from pathlib import Path
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    """
    SAR溢油检测VQA数据集

    返回格式:
        {
            "image": "图片路径",
            "conversations": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    """

    def __init__(self, data_root, split="train"):
        """
        Args:
            data_root: 数据集根目录 (oil_datasets_split)
            split: "train" 或 "test"
        """
        self.data_root = Path(data_root)
        self.split = split

        # 加载VQA数据（用于训练）
        json_path = self.data_root / f"{split}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 加载结构化标签（仅用于评估，不参与训练）
        labels_path = self.data_root / f"{split}_labels.json"
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        # 建立image到label的索引
        self.label_map = {l["image"]: l for l in self.labels}

        print(f"加载 {split} 数据集: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]

        # 验证图片存在
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        return {
            "image": image_path,
            "conversations": item["conversations"]
        }

    def get_label(self, image_path):
        """获取结构化标签（用于评估）"""
        return self.label_map.get(image_path, None)


def collate_fn(batch):
    """自定义collate函数"""
    return {
        "image": [item["image"] for item in batch],
        "conversations": [item["conversations"] for item in batch]
    }
