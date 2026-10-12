import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


BUSI_CLASSES = ["normal", "benign", "malignant"]
BUSI_CLASS2LABEL = {name: i for i, name in enumerate(BUSI_CLASSES)}


def _guess_id_col(df: pd.DataFrame) -> str:
    # 你的 csv 是：Unnamed: 0, ImageID, concept...
    if "ImageID" in df.columns:
        return "ImageID"
    # 兜底：按 skincon 风格，第 2 列一般就是 id
    return df.columns[1]


def _infer_label_from_imageid(image_id: str) -> int:
    """
    你的 ImageID 是：train/benign/xxx.png 或 test/malignant/xxx.png
    这里直接用父文件夹名推断类别。
    """
    parts = Path(image_id).parts
    # 期望 parts[-2] 是 benign/malignant/normal
    if len(parts) >= 2:
        cls = parts[-2].lower()
        if cls in BUSI_CLASS2LABEL:
            return BUSI_CLASS2LABEL[cls]

    # 兜底：用字符串包含判断
    low = image_id.lower()
    if "benign" in low:
        return BUSI_CLASS2LABEL["benign"]
    if "malignant" in low:
        return BUSI_CLASS2LABEL["malignant"]
    if "normal" in low:
        return BUSI_CLASS2LABEL["normal"]

    raise ValueError(f"Cannot infer label from ImageID: {image_id}")


class BUSICSVDataset(Dataset):
    def __init__(self, data_dir: Path, df: pd.DataFrame, transform):
        super().__init__()
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.id_col = _guess_id_col(self.df)

        # 概念列：从第3列开始（Unnamed:0, ImageID, concept...）
        # 这样能完全匹配你当前 csv 的 14 个 concept 名字
        self.concept_cols = list(self.df.columns[2:])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = str(row[self.id_col])  # e.g., "train/benign/benign (1).png"
        img_path = self.data_dir / image_id
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # label 从 ImageID 的父目录推断（benign/malignant/normal）
        y = _infer_label_from_imageid(image_id)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # IMPORTANT: BCELoss 需要 float target
        c = torch.tensor(row[self.concept_cols].astype(float).tolist(), dtype=torch.float32)

        return img, int(y), c


class BUSIDataModule(pl.LightningDataModule):
    """
    训练：用 train.csv 的全部数据（不做 fold）
    验证：为了不浪费训练数据，val 也用 train（但 no-aug，仅用于打印 val 指标）
    测试：用 test.csv 的数据，输出 test/y_acc、test/y_f1 等
    """
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.data_dir = Path(cfgs.data.data_dir)
        self.batch_size = int(cfgs.data.batch_size)

        self.train_csv = Path(getattr(cfgs.data, "train_csv", self.data_dir / "raw_data" / "train.csv"))
        self.test_csv = Path(getattr(cfgs.data, "test_csv", self.data_dir / "raw_data" / "test.csv"))

        self.aug_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(size=cfgs.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize(cfgs.data.img_mean, cfgs.data.img_std),
            ]
        )
        self.noaug_transform = transforms.Compose(
            [
                transforms.Resize(size=cfgs.data.img_size),
                transforms.CenterCrop(size=cfgs.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize(cfgs.data.img_mean, cfgs.data.img_std),
            ]
        )

        # 会在 setup() 时从 csv 自动读出来
        self.concept_list = None
        self.classes = BUSI_CLASSES

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        # 概念列名直接来自 csv（与你当前 train/test.csv 完全一致）
        self.concept_list = list(train_df.columns[2:])

        # 训练：train 全量
        self.train_dataset = BUSICSVDataset(self.data_dir, train_df, self.aug_transform)
        # 验证：不浪费数据 -> 仍用 train，但 no-aug（只是为了打印 val 指标）
        self.valid_dataset = BUSICSVDataset(self.data_dir, train_df, self.noaug_transform)
        # 测试：test 集
        self.test_dataset = BUSICSVDataset(self.data_dir, test_df, self.noaug_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
