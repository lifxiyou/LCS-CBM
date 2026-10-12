# models/crl_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import get_backbones
from models.components import (
    BinarizeLayer,
    LRLayer,
    AndOrLayer,
)
from models.moe import Gate  # 使用你给的 moe.py 里的门控机制


class LogicExpert(nn.Module):
    """
    一个逻辑专家：
    - 输入：二值化后的概念（可能包含 NOT 扩展）
    - 结构：AndOrLayer -> (and_part, or_part) -> concat -> LRLayer -> 分类 logits
    """

    def __init__(self, input_dim, l2_dim, n_classes, use_not=False):
        super().__init__()
        self.use_not = use_not

        # 逻辑组合层：对概念进行组合（与/或/非等）
        self.andor = AndOrLayer(input_dim, l2_dim, use_not=use_not)

        # 假设 and_part 与 or_part 拼接后维度为 512（和你原来的 CRL 一致）
        # 如果你在 AndOrLayer 里改过维度，可以把 512 改成 cfg 里的参数
        self.logic_feature_dim = 512
        self.lr = LRLayer(self.logic_feature_dim, n_classes)

    def forward(self, binary_concepts):
        """
        binary_concepts: (B, D_bin)
        返回:
            logits: 分类 logits, (B, n_classes)
            and_part, or_part: 各自的逻辑特征 (B, ?)
        """
        and_part, or_part = self.andor(binary_concepts, return_split=True)
        combined = torch.cat([and_part, or_part], dim=1)  # (B, 512)
        logits = self.lr(combined)
        return logits, and_part, or_part

    def l2_norm(self):
        """把专家内部的 AndOr 和 LR 的 L2 正则加起来，供主模型使用。"""
        l2 = torch.zeros(1, device=next(self.parameters()).device)
        if hasattr(self.andor, "l2_norm"):
            l2 = l2 + self.andor.l2_norm()
        if hasattr(self.lr, "l2_norm"):
            l2 = l2 + self.lr.l2_norm()
        return l2


class CRL_MoE(pl.LightningModule):
    """
    image -> backbone -> concept predictor -> concept bottleneck
          -> BinarizeLayer -> Gate(门控)
          -> 多个 LogicExpert(And/Or 逻辑组合 + LR 分类)
          -> 门控加权得到最终分类结果
    """

    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()
        self.opt_cfgs = cfgs.opt
        self.l2_weight = cfgs.model.l2_weight
        self.logic_weight = cfgs.model.get("logic_weight", 0.5)

        self.n_concepts = cfgs.data.n_concepts
        self.n_classes = cfgs.data.n_classes
        self.l2_dim = cfgs.model.l2

        # MoE 相关配置
        self.num_experts = getattr(cfgs.model, "num_experts", 4)
        self.capacity_factor = getattr(cfgs.model, "capacity_factor", 1.0)

        # 是否使用 NOT、skip （这里只用 NOT）
        self.use_not = getattr(cfgs.model, "use_not", False)
        print(f"[DEBUG] use_not={self.use_not}, num_experts={self.num_experts}")

        # 温度参数（和原 CRL 一致）
        self.t = nn.Parameter(torch.log(torch.tensor([cfgs.model.temperature])))

        # 预概念 backbone 与概念预测头
        self.pre_concept_model, n_features = get_backbones(cfgs.model.backbone)
        self.concept_predictor = nn.Linear(n_features, self.n_concepts)

        # 概念瓶颈处的二值化层
        self.binarize = BinarizeLayer(self.n_concepts, self.use_not)
        # 尽量从 BinarizeLayer 中读取输出维度；否则按 2x 做一个保守假设
        binary_dim = getattr(self.binarize, "output_dim",
                             self.n_concepts * (2 if self.use_not else 1))

        # 门控：在概念瓶颈处对多专家进行路由
        self.gate = Gate(
            dim=binary_dim,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor,
        )

        # 多个逻辑专家：每个专家各自有一套 And/Or 组合 + 线性分类器
        self.logic_experts = nn.ModuleList([
            LogicExpert(
                input_dim=binary_dim,
                l2_dim=self.l2_dim,
                n_classes=self.n_classes,
                use_not=self.use_not,
            )
            for _ in range(self.num_experts)
        ])

        # 损失函数（概念监督 + 分类）
        self.loss_concept = nn.BCELoss()
        self.loss_cls = nn.CrossEntropyLoss()

        # 每一步 forward 中记录 gate 的辅助 loss
        self.current_gate_loss = None

    # --------- 正则项 ---------
    def l2_penalty(self):
        l2_penalty = torch.zeros(1, device=self.device)
        for expert in self.logic_experts:
            if hasattr(expert, "l2_norm"):
                l2_penalty = l2_penalty + expert.l2_norm()
        return l2_penalty

    # --------- 前向传播 ---------
    def forward(self, x):
        """
        返回：
            y_out: (B, n_classes)
            concepts: (B, n_concepts)  未 sigmoid 的概念 logits
            and_agg: 聚合后的 and 部分特征 (B, D_and)
            or_agg: 聚合后的 or 部分特征 (B, D_or)
        """
        # 图像 -> backbone -> 特征 -> 概念 logits
        features = self.pre_concept_model(x)
        concepts = self.concept_predictor(features)  # (B, n_concepts)

        # 概念瓶颈：二值化（支持 NOT）
        binary_concepts = self.binarize(concepts)  # (B, D_bin)

        # ---- 门控：根据二值化概念选择不同逻辑专家 ----
        gate_scores, gate_loss = self.gate(binary_concepts, use_aux_loss=True)
        self.current_gate_loss = gate_loss

        # 多专家逻辑组合
        logits_list = []
        and_list = []
        or_list = []

        for expert in self.logic_experts:
            logits_i, and_i, or_i = expert(binary_concepts)
            logits_list.append(logits_i)   # (B, n_classes)
            and_list.append(and_i)        # (B, D_and)
            or_list.append(or_i)          # (B, D_or)

        # stack: (B, n_experts, *)
        logits_stack = torch.stack(logits_list, dim=1)  # (B, E, n_classes)
        and_stack = torch.stack(and_list, dim=1)        # (B, E, D_and)
        or_stack = torch.stack(or_list, dim=1)          # (B, E, D_or)

        # 防止 gate_scores NaN
        if torch.isnan(gate_scores).any():
            print("[WARNING] NaN in gate_scores, replacing with zeros.")
            gate_scores = gate_scores.clone()
            gate_scores[torch.isnan(gate_scores)] = 0.0

        # gate_scores: (B, E) -> (B, E, 1)
        gate_scores_expanded = gate_scores.unsqueeze(-1)

        # 门控加权得到最终分类 logits
        y_out = torch.sum(gate_scores_expanded * logits_stack, dim=1)  # (B, n_classes)

        # 门控加权后的逻辑特征，方便可解释性可视化
        and_agg = torch.sum(gate_scores_expanded * and_stack, dim=1)   # (B, D_and)
        or_agg = torch.sum(gate_scores_expanded * or_stack, dim=1)     # (B, D_or)

        return y_out, concepts, and_agg, or_agg

    # --------- 训练步骤 ---------
    def training_step(self, batch, _):
        x, y, c = batch
        y_bar, c_bar, and_part, or_part = self.forward(x)

        # 温度缩放 + sigmoid
        y_bar = y_bar / torch.exp(self.t)
        c_bar_prob = torch.sigmoid(c_bar)

        # 概念监督（和你当前 CRL 一样，只是作为监控日志，不加到最终 loss 里）
        loss_list = []
        for i in range(c.shape[1]):
            single_c = c[:, i].float()
            pred = c_bar_prob[:, i]
            loss_list.append(self.loss_concept(pred, single_c))
        combined_loss = sum(loss_list) / len(loss_list)

        # 分类 loss
        rrl_loss = self.loss_cls(y_bar, y)

        # L2 正则（所有逻辑专家）
        l2_loss = self.l2_weight * self.l2_penalty()

        # 门控的均衡 loss（load balancing），来自 Gate / moe.py
        gate_loss = self.current_gate_loss
        if gate_loss is None:
            gate_loss = torch.zeros(1, device=self.device)
        logic_loss = self.logic_weight * gate_loss

        # 总 loss：分类 + L2 正则 + 门控辅助 loss
        loss = rrl_loss + l2_loss + logic_loss

        self.log("combined_loss", combined_loss)
        self.log("l2_loss", l2_loss)
        self.log("rrl_loss", rrl_loss)
        self.log("gate_loss", gate_loss)
        self.log("logic_loss", logic_loss)
        self.log("loss", loss)

        return loss

    # --------- 验证 / 测试 / 预测 ---------
    def validation_step(self, batch, _):
        x, _, _ = batch
        y_out, concepts, _, _ = self.forward(x)
        concepts = torch.sigmoid(concepts)
        return {"y": y_out, "c": concepts}

    def test_step(self, batch, _):
        x, _, _ = batch
        y_out, concepts, _, _ = self.forward(x)
        concepts = torch.sigmoid(concepts)
        return {"y": y_out, "c": concepts}

    def predict_step(self, batch, _):
        x, _, _ = batch
        y_out, concepts, _, _ = self.forward(x)
        concepts = torch.sigmoid(concepts)
        return {"y": y_out, "c": concepts}

    # --------- 优化器配置（和原 CRL 保持一致） ---------
    def configure_optimizers(self):
        param_groups = [
            {
                "params": [param for name, param in self.named_parameters() if "concept" in name],
                "lr": self.opt_cfgs.lr / 100,
                "weight_decay": 1e-2,
            },
            {
                "params": [param for name, param in self.named_parameters() if "concept" not in name],
                "weight_decay": 0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.opt_cfgs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.opt_cfgs.max_epochs)
        return [optimizer], [scheduler]
