import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class CMMDDataset(Dataset):
    def __init__(self, data_dir: Path, df: pd.DataFrame, transform):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Must match folder names in your CMMD dataset
        self.classes = ["benign", "malignant"]

        self.concept_list = df.columns[2:].tolist()
        if "Do not consider this image" in self.concept_list:
            self.concept_list.remove("Do not consider this image")

    @staticmethod
    def convert_image_to_rgb(image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        rel_path = str(sample["ImageID"]).replace("\\", "/")
        img_path = self.data_dir / rel_path

        image = Image.open(img_path)
        image = self.convert_image_to_rgb(image)
        image = self.transform(image)

        # label inferred from path: "train/benign/xxx.png" -> "benign"
        parts = rel_path.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected ImageID format: {rel_path}")
        cls_name = parts[1]
        if cls_name not in self.classes:
            raise ValueError(f"Unknown class folder '{cls_name}' in ImageID: {rel_path}")
        label = self.classes.index(cls_name)

        concepts = torch.FloatTensor(sample.loc[self.concept_list].tolist())
        return image, int(label), concepts


class CMMDDataModule(pl.LightningDataModule):
    """
    CMMD datamodule for CRL.

    - train_dataloader(): uses train.csv with augmentation
    - val_dataloader(): uses the SAME train.csv without augmentation (so val metrics reflect train set)
    - test_dataloader(): uses test.csv without augmentation
    """

    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.data_dir = Path(cfgs.data.data_dir)
        self.batch_size = int(cfgs.data.batch_size)
        self.num_workers = int(getattr(cfgs.data, "num_workers", 4))

        # CSV paths (prefer explicit cfgs; fallback to meta_data/)
        self.train_csv = getattr(cfgs.data, "train_csv", str(self.data_dir / "meta_data" / "train.csv"))
        self.test_csv = getattr(cfgs.data, "test_csv", str(self.data_dir / "meta_data" / "test.csv"))

        # Read once to obtain concept names
        df = pd.read_csv(self.train_csv)
        self.concept_list = df.columns[2:].tolist()
        if "Do not consider this image" in self.concept_list:
            self.concept_list.remove("Do not consider this image")

        self.classes = ["benign", "malignant"]

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

        self.train_dataset = CMMDDataset(self.data_dir, train_df, self.aug_transform)
        # "val" uses train data (no-aug) to avoid holding out data
        self.valid_dataset = CMMDDataset(self.data_dir, train_df, self.noaug_transform)
        self.test_dataset = CMMDDataset(self.data_dir, test_df, self.noaug_transform)

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
        # used for rule extraction; keep consistent with train set
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
