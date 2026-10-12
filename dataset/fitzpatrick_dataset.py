"""
Fitzpatrick17k 数据集，供黑盒模型训练/验证/测试使用。
使用 meta_data/5_fold_splits.json，flag 表示折下标；只返回 (image, label)，无 concept。
"""
import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Fitzpatrick17kDataset(Dataset):
    """Fitzpatrick17k：按 5_fold_splits.json 的某一折加载 train/valid，只返回图像与分类标签。"""

    def __init__(self, data_path, mode='train', transforms=None, flag=0, debug=False, config=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms
        self.flag = int(flag)
        self.config = config

        json_path = os.path.join(data_path, "meta_data", "5_fold_splits.json")
        with open(json_path, "r") as f:
            splits = json.load(f)

        key = f"fold_{self.flag}"
        if key not in splits:
            raise ValueError(f"Fold {key} not found in {json_path}")

        # Fitzpatrick17k 只有 train / valid，没有 test；val 和 test 都用 valid
        if mode == "train":
            split_name = "train"
        else:
            split_name = "valid"

        if split_name not in splits[key]:
            raise ValueError(f"Split '{split_name}' not found for {key}")

        items = splits[key][split_name]
        self.images = []   # 存 img_id（文件名）
        self.labels = []

        for item in items:
            img_id = item["id"]
            label = item["label"]
            img_path = os.path.join(data_path, "raw_data", img_id)
            if not os.path.exists(img_path):
                continue
            self.images.append(img_id)
            self.labels.append(label)

        if debug and len(self.images) > 200:
            self.images = self.images[:200]
            self.labels = self.labels[:200]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found for Fitzpatrick17k {key} {split_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_id = self.images[index]
        label = self.labels[index]
        img_path = os.path.join(self.data_path, "raw_data", img_id)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label
