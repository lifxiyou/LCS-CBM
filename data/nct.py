import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


# Folder names under train/ and test/ (must match exactly)
NCT_CLASS_FOLDERS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


class NCTCSVDataset(Dataset):
    def __init__(self, data_dir: Path, df: pd.DataFrame, transform):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.classes = NCT_CLASS_FOLDERS
        self.id_col = "ImageID" if "ImageID" in df.columns else df.columns[1]
        self.concept_list = df.columns[2:].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        rel_path = str(row[self.id_col]).replace("\\", "/")
        img_path = self.data_dir / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        parts = rel_path.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected ImageID format: {rel_path}")
        cls_name = parts[1]
        if cls_name not in self.classes:
            raise ValueError(f"Unknown class folder '{cls_name}' in ImageID: {rel_path}")
        y = self.classes.index(cls_name)

        c = torch.tensor(row[self.concept_list].astype(float).tolist(), dtype=torch.float32)
        return image, int(y), c


class NCTDataModule(pl.LightningDataModule):
    """NCT datamodule for CRL.

    - train_dataloader(): uses train.csv with augmentation
    - val_dataloader(): uses the SAME train.csv without augmentation (val metrics reflect train set)
    - test_dataloader(): uses test.csv without augmentation
    """

    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.data_dir = Path(cfgs.data.data_dir)
        self.batch_size = int(cfgs.data.batch_size)
        self.num_workers = int(getattr(cfgs.data, "num_workers", 4))

        self.train_csv = getattr(cfgs.data, "train_csv", str(self.data_dir / "meta_data" / "train.csv"))
        self.test_csv = getattr(cfgs.data, "test_csv", str(self.data_dir / "meta_data" / "test.csv"))

        df = pd.read_csv(self.train_csv)
        self.concept_list = df.columns[2:].tolist()
        self.classes = NCT_CLASS_FOLDERS

        img_size = int(cfgs.data.img_size)

        self.aug_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(cfgs.data.img_mean, cfgs.data.img_std),
            ]
        )
        self.noaug_transform = transforms.Compose(
            [
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(cfgs.data.img_mean, cfgs.data.img_std),
            ]
        )

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        self.train_dataset = NCTCSVDataset(self.data_dir, train_df, self.aug_transform)
        self.valid_dataset = NCTCSVDataset(self.data_dir, train_df, self.noaug_transform)
        self.test_dataset = NCTCSVDataset(self.data_dir, test_df, self.noaug_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
