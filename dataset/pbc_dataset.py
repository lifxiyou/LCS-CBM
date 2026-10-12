"""
PBC 数据集，供黑盒模型训练/验证/测试使用。
使用 meta_data/train.csv、valid.csv、test.csv，只返回 (image, label)，无 concept。
"""
import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PBCDataset(Dataset):
    """PBC：按 train/valid/test.csv 加载，只返回图像与分类标签。类别顺序以 train.csv 的 sorted(unique(label)) 为准。"""

    def __init__(self, data_path, mode='train', transforms=None, flag=0, debug=False, config=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms
        self.flag = int(flag)
        self.config = config

        # 用 train.csv 确定全局类别顺序，保证 train/val/test 一致
        train_csv = os.path.join(data_path, "meta_data", "train.csv")
        with open(train_csv, "r") as f:
            reader = csv.DictReader(f)
            all_labels = [row["label"] for row in reader]
        self.class_names = sorted(list(set(all_labels)))
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 当前划分
        if mode == "train":
            csv_name = "train.csv"
        elif mode == "val":
            csv_name = "valid.csv"
            if not os.path.exists(os.path.join(data_path, "meta_data", "valid.csv")):
                csv_name = "train.csv"
        else:
            csv_name = "test.csv"

        csv_path = os.path.join(data_path, "meta_data", csv_name)
        if not os.path.exists(csv_path):
            raise ValueError(f"PBC meta file not found: {csv_path}")

        self.images = []
        self.labels = []
        prefix = "PBC_dataset_normal_DIB/"

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["path"].strip()
                if rel_path.startswith(prefix):
                    rel_path = rel_path[len(prefix):]
                img_path = os.path.join(data_path, "raw_data", rel_path)
                if not os.path.exists(img_path):
                    continue
                label_name = row["label"].strip()
                if label_name not in self.label_to_idx:
                    continue
                self.images.append(img_path)
                self.labels.append(self.label_to_idx[label_name])

        if debug and len(self.images) > 500:
            self.images = self.images[:500]
            self.labels = self.labels[:500]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found for PBC {mode} in {csv_path}")

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
