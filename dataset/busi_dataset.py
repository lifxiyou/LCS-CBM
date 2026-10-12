"""
BUSI 数据集，供黑盒模型训练/验证/测试使用。
使用 meta_data/train.csv、test.csv，ImageID 形如 "train/benign/benign (1).png"，标签从路径第二段得到。
只返回 (image, label)，无 concept。
"""
import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    """BUSI：按 train.csv / test.csv 加载，只返回图像与分类标签。类别顺序以 train 的 sorted(unique) 为准。"""

    def __init__(self, data_path, mode='train', transforms=None, flag=0, debug=False, config=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms
        self.flag = int(flag)
        self.config = config

        # 用 train.csv 确定全局类别顺序
        train_csv = os.path.join(data_path, "meta_data", "train.csv")
        with open(train_csv, "r") as f:
            reader = csv.DictReader(f)
            all_labels = []
            for row in reader:
                rel_path = row["ImageID"].strip().replace("\\", "/")
                parts = rel_path.split("/")
                if len(parts) >= 2:
                    all_labels.append(parts[1].lower())
        self.class_names = sorted(list(set(all_labels)))
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}

        if mode == "val":
            csv_name = "train.csv"
        elif mode == "test":
            csv_name = "test.csv"
        else:
            csv_name = "train.csv"

        csv_path = os.path.join(data_path, "meta_data", csv_name)
        if not os.path.exists(csv_path):
            raise ValueError(f"BUSI meta file not found: {csv_path}")

        self.images = []
        self.labels = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["ImageID"].strip().replace("\\", "/")
                img_path = os.path.join(data_path, rel_path)
                if not os.path.exists(img_path):
                    continue
                parts = rel_path.split("/")
                if len(parts) < 2:
                    continue
                label_str = parts[1].lower()
                if label_str not in self.label_to_idx:
                    continue
                self.images.append(img_path)
                self.labels.append(self.label_to_idx[label_str])

        if debug and len(self.images) > 500:
            self.images = self.images[:500]
            self.labels = self.labels[:500]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found for BUSI {mode} in {csv_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label
