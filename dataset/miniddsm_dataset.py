"""
MiniDDSM 数据集，供黑盒模型训练/验证/测试使用。
使用 meta_data/train.csv、test.csv，ImageID 形如 "train/benign/xxx.png" 或 "test/malignant/yyy.png"，
标签从路径第二段得到（benign / malignant）。只返回 (image, label)，无 concept。
"""
import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 与 data/miniddsm.py 一致
MINIDDSM_CLASS_FOLDERS = ["benign", "malignant"]


class MiniDDSMDataset(Dataset):
    """MiniDDSM：按 train.csv / test.csv 加载，只返回图像与分类标签。类别顺序 benign=0, malignant=1。"""

    def __init__(self, data_path, mode='train', transforms=None, flag=0, debug=False, config=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms
        self.flag = int(flag)
        self.config = config
        self.class_names = MINIDDSM_CLASS_FOLDERS
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}

        if mode == "val":
            csv_name = "train.csv"
        elif mode == "test":
            csv_name = "test.csv"
        else:
            csv_name = "train.csv"

        csv_path = os.path.join(data_path, "meta_data", csv_name)
        if not os.path.exists(csv_path):
            raise ValueError(f"MiniDDSM meta file not found: {csv_path}")

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
            raise RuntimeError(f"No images found for MiniDDSM {mode} in {csv_path}")

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
