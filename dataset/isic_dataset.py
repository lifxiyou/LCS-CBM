"""
ISIC2018 数据集，供黑盒模型训练/验证/测试使用。
接口与 Fitzpatrick17k 一致：(data_path, mode, transforms, flag, debug, config)，只返回 (image, label)。
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# 与 data/isic2018.py 保持一致
ISIC2018_CLASS_FOLDERS = [
    "a vascular lesion",
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanocytic nevus",
    "melanoma",
]


class SkinDataset(Dataset):
    """ISIC2018：从 train.csv / test.csv 加载，只返回 (image, label)，供黑盒训练。"""

    def __init__(self, data_path, mode='train', transforms=None, flag=0, debug=False, config=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.mode = mode
        self.transforms = transforms
        self.flag = int(flag)
        self.config = config
        self.classes = ISIC2018_CLASS_FOLDERS

        train_csv = self.data_path / "meta_data" / "train.csv"
        test_csv = self.data_path / "meta_data" / "test.csv"
        if not train_csv.exists():
            train_csv = getattr(config, "train_csv", train_csv)
        if not test_csv.exists():
            test_csv = getattr(config, "test_csv", test_csv)

        if mode == "test":
            self.df = pd.read_csv(test_csv)
        else:
            self.df = pd.read_csv(train_csv)

        self.id_col = "ImageID" if "ImageID" in self.df.columns else self.df.columns[1]
        if debug and len(self.df) > 500:
            self.df = self.df.iloc[:500].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        rel_path = str(row[self.id_col]).replace("\\", "/")
        img_path = self.data_path / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        parts = rel_path.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected ImageID format: {rel_path}")
        cls_name = parts[1]
        if cls_name not in self.classes:
            raise ValueError(f"Unknown class folder '{cls_name}' in {rel_path}")
        y = self.classes.index(cls_name)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, int(y)
