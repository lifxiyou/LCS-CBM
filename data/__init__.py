from .skincon import SkinConDataModule
from .wbcatt import WBCAttDataModule
from .busi import BUSIDataModule
from .cmmd import CMMDDataModule
from .idrid import IDRiDDataModule
from .isic2018 import ISIC2018DataModule
from .miniddsm import MiniDDSMDataModule
from .nct import NCTDataModule
from .siim import SIIMDataModule





def get_datamodules(cfgs):
    if cfgs.data.dataset == "skincon":
        datamodule = SkinConDataModule(cfgs)
    elif cfgs.data.dataset == "wbcatt":
        datamodule = WBCAttDataModule(cfgs)
    elif cfgs.data.dataset == "busi":
        datamodule = BUSIDataModule(cfgs)
    elif cfgs.data.dataset == "cmmd":
        datamodule = CMMDDataModule(cfgs)
    elif cfgs.data.dataset == "idrid":
        datamodule = IDRiDDataModule(cfgs)
    elif cfgs.data.dataset == "isic2018":
        datamodule = ISIC2018DataModule(cfgs)
    elif cfgs.data.dataset == "miniddsm":
        datamodule = MiniDDSMDataModule(cfgs)
    elif cfgs.data.dataset == "nct":
        datamodule = NCTDataModule(cfgs)
    elif cfgs.data.dataset == "siim":
        datamodule = SIIMDataModule(cfgs)
    else:
        raise NotImplementedError
    return datamodule
