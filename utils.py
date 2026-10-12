import torch
import wandb
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.loggers import CSVLogger, WandbLogger


def get_interpolation_mode(interpolation):
    """将 timm 的 interpolation 配置（int 或 str 如 'bicubic'）转为 torchvision 可用的插值模式。"""
    if hasattr(transforms, "InterpolationMode"):
        mode_map_int = {
            0: transforms.InterpolationMode.NEAREST,
            1: transforms.InterpolationMode.BILINEAR,
            2: transforms.InterpolationMode.BICUBIC,
            3: transforms.InterpolationMode.LANCZOS,
        }
        mode_map_str = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }
        if isinstance(interpolation, str):
            return mode_map_str.get(interpolation.lower(), transforms.InterpolationMode.BILINEAR)
        try:
            return mode_map_int.get(int(interpolation), transforms.InterpolationMode.BILINEAR)
        except (TypeError, ValueError):
            return transforms.InterpolationMode.BILINEAR
    import PIL.Image
    return PIL.Image.BILINEAR


def gray_world():
    """Gray-world 颜色校正（用于皮肤图像），作为可调用的 transform。"""
    class GrayWorld(object):
        def __call__(self, tensor):
            return tensor  # 占位，保持与原有 pipeline 兼容
    return GrayWorld()


def identity():
    """不做颜色校正的占位 transform。"""
    class Identity(object):
        def __call__(self, tensor):
            return tensor
    return Identity()


def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch, gamma=0.1):
    """
    每 epoch 调用：前 warmup_epoch 轮线性从 0 升到 init_lr，之后按指数衰减。
    optimizer: 优化器（会原地修改 param_groups 的 lr）
    init_lr: 目标初始学习率
    epoch: 当前 epoch（0-based）
    warmup_epoch: warmup 轮数
    max_epoch: 总轮数
    gamma: 衰减系数，最终 lr ≈ init_lr * gamma
    """
    if epoch < warmup_epoch:
        lr = init_lr * (epoch + 1) / warmup_epoch
    else:
        # 指数衰减：从 warmup 结束到 max_epoch，lr 从 init_lr 衰减到 init_lr * gamma
        progress = (epoch - warmup_epoch) / max(1, max_epoch - warmup_epoch)
        lr = init_lr * (gamma ** progress)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def setup_logger(cfgs):
    pl.seed_everything(cfgs.seed)

    Path(cfgs.log.save_dir).mkdir(parents=True, exist_ok=True)

    if cfgs.log.logger == "wandb":
        logger = WandbLogger(
            cfgs.exp_name,
            cfgs.log.save_dir,
            project=cfgs.project_name,
            config=cfgs,
            settings=wandb.Settings(code_dir="."),
        )
    elif cfgs.log.logger == "csv":
        logger = CSVLogger(cfgs.log.save_dir, cfgs.exp_name)
    else:
        logger = False
    return logger


def get_backbones(name, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    model = getattr(torchvision.models, name)(weights=weights)
    if name.startswith("resnet"):
        n_features = list(model.modules())[-1].in_features
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported image encoder: {name}")
    return model, n_features
