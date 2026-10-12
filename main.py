import pytorch_lightning as pl

from models import CRL
# from models.CNSCRL_MoE import CNSCRL_MoE
from models.cns_crl import CNSCRL
from models.CNSCRL_MoE import CNSCRL_MoE
# from models.crl_moe import CRL_MoE
from utils import setup_logger
from configs import get_configs
from data import get_datamodules
from callbacks import get_callbacks


def main(cfgs):
    logger = setup_logger(cfgs)
    model = CRL(cfgs)
    datamodule = get_datamodules(cfgs)
    callbacks = get_callbacks(cfgs)
    # import pdb; pdb.set_trace()

    # 🔹 在这里添加打印逻辑层结构的代码
    print("\n=== Model Layer Outputs ===")
    try:
        layer_list = model.layer_list  # 注意：layer_list 可能是 @property，取值可能抛异常
    except Exception as e:
        layer_list = None
        print(f"跳过 layer_list 打印（获取 layer_list 失败）：{type(e).__name__}: {e}")

    if layer_list is not None:
        for i, layer in enumerate(layer_list):
            layer_type = getattr(layer, "layer_type", layer.__class__.__name__)
            prev = (
                getattr(layer.conn.prev_layer, "layer_type", "None")
                if hasattr(layer, "conn") and layer.conn.prev_layer
                else "None"
            )
            skip = (
                getattr(layer.conn.skip_from_layer, "layer_type", "None")
                if hasattr(layer, "conn") and layer.conn.skip_from_layer
                else "None"
            )
            out_dim = getattr(layer, "output_dim", getattr(layer, "out_features", "N/A"))
            print(f"[{i}] {layer_type:<20} | prev={prev:<15} | skip={skip:<15} | out_dim={out_dim}")
    print("============================\n")


  


    trainer = pl.Trainer(
        devices=[cfgs.device],
        logger=logger,
        max_epochs=cfgs.opt.max_epochs,
        log_every_n_steps=cfgs.log.log_every_n_steps,
        callbacks=callbacks,
        check_val_every_n_epoch=cfgs.log.val_every_n_epoch,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    datamodule.setup("fit")
    print("TRAIN:", len(datamodule.train_dataloader().dataset))
    print("VAL:", len(datamodule.val_dataloader().dataset))

    datamodule.setup("test")
    print("TEST:", len(datamodule.test_dataloader().dataset))

    trainer.test(model, datamodule=datamodule)

    # extract rules
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    cfgs = get_configs()
    main(cfgs)
