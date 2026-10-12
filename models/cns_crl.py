import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import get_backbones
from models.components import Connection


# ----------------------
# Soft binarizer
# ----------------------
class SoftBinarizeLayer(nn.Module):
    def __init__(self, dim, use_not=True, init_temp=0.5):
        super().__init__()
        self.use_not = use_not
        self.t = nn.Parameter(torch.tensor(init_temp))
        self.output_dim = dim * 2 if use_not else dim
        self.bl = torch.zeros(self.output_dim)
        self.rule_name = "SoftBinarize"  # 新增：占位符名称

    def forward(self, logits):
        probs = torch.sigmoid(logits / self.t.clamp(min=1e-3))
        if self.use_not:
            probs = torch.cat([probs, 1 - probs], dim=-1)
        return probs

    def binarized_forward(self, logits):
        probs = torch.sigmoid(logits / self.t.clamp(min=1e-3))
        if self.use_not:
            probs = torch.cat([probs, 1 - probs], dim=-1)
        return (probs > 0.5).float()

    def clip(self):
        # 占位方法，保持和 CRL 的接口一致
        pass

    # 🔥 新增方法
    def get_rule_name(self, feature_name=None):
        """
        返回每个概念对应的名字（正向和取反）
        feature_name: 可选list[str]，概念的实际名称
        """
        n = self.output_dim // (2 if self.use_not else 1)
        names = []
        for i in range(n):
            if feature_name is not None and i < len(feature_name):
                base = feature_name[i]
            else:
                base = f"C{i}"
            names.append(base)
            if self.use_not:
                names.append(f"NOT {base}")
        return names
    
    def get_rule2weights(self, prev_layer=None, skip_from_layer=None):
        weights = {}


# ----------------------
# Neural-Symbolic Block
# ----------------------
# 1e-4
class NeuralSymbolicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, and_or_ratio=0.5, l1_lambda=1e-6):
        super().__init__()
        # self.num_and = int(out_dim * and_or_ratio)
        self.num_and = out_dim              # 全部都是 AND
        self.num_or = 0                     # out_dim - self.num_and
        self.gates_and = nn.Parameter(torch.randn(self.num_and, in_dim))
        # self.gates_or = nn.Parameter(torch.randn(self.num_or, in_dim))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.l1_lambda = l1_lambda
        self.output_dim = out_dim
        self.rule_name = f"NSBlock({out_dim})"  # 新增：占位符名称

    @staticmethod
    def l_and(x, y):  # fuzzy AND
        return torch.clamp(x + y - 1, min=0.0)

    @staticmethod
    def l_or(x, y):  # fuzzy OR
        return torch.clamp(x + y, max=1.0)

    def _aggregate(self, x, gates, mode="and"):
        B, D = x.shape
        K, _ = gates.shape
        sel = torch.sigmoid(gates / self.temp.clamp(min=1e-3))  # [K, D]
        xv = x.unsqueeze(1) * sel.unsqueeze(0)  # [B,K,D]
        out = xv[..., 0]
        for d in range(1, D):
            out = self.l_and(out, xv[..., d]) if mode == "and" else self.l_or(out, xv[..., d])
        return out  # [B,K]

    def forward(self, x):
        # outs = []
        # if self.num_and > 0:
        #     outs.append(self._aggregate(x, self.gates_and, "and"))
        # if self.num_or > 0:
        #     outs.append(self._aggregate(x, self.gates_or, "or"))
        # return torch.cat(outs, dim=-1)
        return self._aggregate(x, self.gates_and, "and")


    # def sparsity_loss(self):
    #     return self.l1_lambda * (self.gates_and.abs().sum() + self.gates_or.abs().sum())

    # def clip(self):
    #     # 保持和 CRL 接口一致，可以在这里做参数约束
    #     self.gates_and.data.clamp_(-5, 5)
    #     self.gates_or.data.clamp_(-5, 5)

    def sparsity_loss(self):
        return self.l1_lambda * self.gates_and.abs().sum()

    def clip(self):
        self.gates_and.data.clamp_(-5, 5)


    # def get_rules(self, prev_layer=None, skip_from_layer=None, feature_names=None):
    #     """
    #     将 gates 转换成规则表达式（占位版本）
    #     prev_layer / skip_from_layer: 用于残差结构的上层引用
    #     feature_names: 概念名列表
    #     """
    #     rules = []
    #     # ... 生成 AND 和 OR 的规则 ...
    #     self.rule_name = rules   # 把规则保存下来，供 callbacks 用
    #     return rules

    #     # AND 规则
    #     if self.num_and > 0:
    #         sel_and = torch.sigmoid(self.gates_and).detach().cpu().numpy()
    #         for i, row in enumerate(sel_and):
    #             active_idx = [j for j, v in enumerate(row) if v > 0.5]
    #             if feature_names:
    #                 terms = [feature_names[j] for j in active_idx]
    #             else:
    #                 terms = [f"C{j}" for j in active_idx]
    #             rules.append(" AND ".join(terms) if terms else "TRUE")

    #     # OR 规则
    #     if self.num_or > 0:
    #         sel_or = torch.sigmoid(self.gates_or).detach().cpu().numpy()
    #         for i, row in enumerate(sel_or):
    #             active_idx = [j for j, v in enumerate(row) if v > 0.5]
    #             if feature_names:
    #                 terms = [feature_names[j] for j in active_idx]
    #             else:
    #                 terms = [f"C{j}" for j in active_idx]
    #             rules.append(" OR ".join(terms) if terms else "FALSE")

    #     return rules

    def get_rules(self, prev_layer=None, skip_from_layer=None, feature_names=None):
        rules = []
        sel_and = torch.sigmoid(self.gates_and).detach().cpu().numpy()
        for i, row in enumerate(sel_and):
            active_idx = [j for j, v in enumerate(row) if v > 0.5]
            if feature_names:
                terms = [feature_names[j] for j in active_idx]
            else:
                terms = [f"C{j}" for j in active_idx]
            rules.append(" AND ".join(terms) if terms else "TRUE")
        self.rule_name = rules
        return rules


    def get_rule2weights(self, prev_layer=None, skip_from_layer=None):
        weights = {}
        w_and = torch.sigmoid(self.gates_and).detach().cpu().numpy()
        for i, row in enumerate(w_and):
            weights[f"AND_{i}"] = row.tolist()
        return weights

    
    def get_rule_description(self, prev_layer=None, skip_from_layer=None, feature_names=None, wrap=False):
        rules = self.get_rules(prev_layer, skip_from_layer, feature_names)
        if wrap:
            return [f"({r})" for r in rules]
        return rules

    
    # def get_rule2weights(self, prev_layer=None, skip_from_layer=None):
    #     weights = {}

    #     if self.num_and > 0:
    #         w_and = torch.sigmoid(self.gates_and).detach().cpu().numpy()
    #         for i, row in enumerate(w_and):
    #             weights[f"AND_{i}"] = row.tolist()

    #     if self.num_or > 0:
    #         w_or = torch.sigmoid(self.gates_or).detach().cpu().numpy()
    #         for i, row in enumerate(w_or):
    #             weights[f"OR_{i}"] = row.tolist()

    #     return weights

# ----------------------
# CNS-CRL model
# ----------------------
class CNSCRL(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()
        self.opt_cfgs = cfgs.opt

        # backbone + concept predictor
        self.pre_concept_model, n_features = get_backbones(cfgs.model.backbone)
        self.concept_predictor = nn.Linear(n_features, cfgs.data.n_concepts)

        # soft binarizer
        self.soft_bin = SoftBinarizeLayer(cfgs.data.n_concepts, use_not=cfgs.model.use_not)

        # multi-layer logic
        self.logic1 = NeuralSymbolicBlock(self.soft_bin.output_dim, cfgs.model.l1)
        self.logic2 = NeuralSymbolicBlock(self.logic1.output_dim, cfgs.model.l2)

        # classifier
        self.classifier = LinearWithRules(self.logic2.output_dim, cfgs.data.n_classes)
        # self.classifier = MLPWithRules(
        #     self.logic2.output_dim, 128, cfgs.data.n_classes
        # )


        # residual 开关
        self.use_residual = getattr(cfgs.model, "use_residual", False)

        # losses
        self.loss_concept = nn.BCEWithLogitsLoss()
        self.loss_cls = nn.CrossEntropyLoss()
        self.cf_weight = getattr(cfgs.model, "cf_weight", 0.1)

        # print(f"[DEBUG] logic1 output_dim = {self.logic1.output_dim}, logic2 output_dim = {self.logic2.output_dim}")


        # 兼容 CRL 的 layer_list
        self.layer_list = nn.ModuleList([
            self.soft_bin,
            self.logic1,
            self.logic2,
            self.classifier,
        ])
        # 给每一层加上连接信息，兼容 CRL 的残差结构
        prev_layer = None
        for i, layer in enumerate(self.layer_list):
            skip_from_layer = None
            if getattr(cfgs.model, "use_skip", False) and i >= 2:
                # 比如 logic2 可以接收来自 soft_bin 的 skip
                skip_from_layer = self.layer_list[i - 2]

            layer.conn = Connection(
                prev_layer=prev_layer,
                is_skip_to_layer=False,
                skip_from_layer=skip_from_layer,
            )
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True

            prev_layer = layer

    def forward(self, x, return_concepts=False):
        feats = self.pre_concept_model(x)
        c_logits = self.concept_predictor(feats)  # [B,C]
        c_soft = self.soft_bin(c_logits)

        z1 = self.logic1(c_soft)

        z2 = self.logic2(z1)
        if self.use_residual:
            # 残差连接：把 z1 加回来
            z2 = z2 + z1

        y = self.classifier(z2)
        if return_concepts:
            return y, c_logits, c_soft
        return y


    def bi_forward(self, x):
        feats = self.pre_concept_model(x)
        c_logits = self.concept_predictor(feats)
        c_bin = self.soft_bin.binarized_forward(c_logits)

        z1 = self.logic1(c_bin)
        z2 = self.logic2(z1)
        if self.use_residual:
            z2 = z2 + z1

        y = self.classifier(z2)
        return y, c_bin


    # 反事实生成
    def random_counterfactual(self, c_soft, flip_ratio=0.2):
        B, D = c_soft.shape
        k = max(1, int(D * flip_ratio))
        idx = torch.randperm(D, device=c_soft.device)[:k]
        c_cf = c_soft.clone()
        c_cf[:, idx] = 1 - c_cf[:, idx]
        return c_cf

    def training_step(self, batch, _):
        x, y, c = batch
        c = c.float()
        y = y.long()

        y_hat, c_logits, c_soft = self.forward(x, return_concepts=True)

        # 损失
        loss_c = self.loss_concept(c_logits, c)
        loss_y = self.loss_cls(y_hat, y)
        sparse_loss = self.logic1.sparsity_loss() + self.logic2.sparsity_loss()

        # 反事实一致性
        with torch.no_grad():
            c_cf = self.random_counterfactual(c_soft)
        z1_cf = self.logic1(c_cf)
        z2_cf = self.logic2(z1_cf)
        y_cf = self.classifier(z2_cf)

        # 用 KL 散度替换 MSE
        cf_loss = F.kl_div(
            F.log_softmax(y_cf, dim=-1),
            F.softmax(y_hat, dim=-1),
            reduction="batchmean"
        )


        loss = loss_c + loss_y  + sparse_loss  + self.cf_weight * cf_loss
        # # 在 training_step 内部
        # progress = self.current_epoch / max(1, self.opt_cfgs.max_epochs)
        # cf_w = self.cf_weight * min(1.0, progress * 2.0)  # 前半阶段逐渐增大
        # loss = loss_c + loss_y + sparse_loss + cf_w * cf_loss


        self.log("loss", loss)
        self.log("concept_loss", loss_c)
        self.log("cls_loss", loss_y)
        self.log("sparse_loss", sparse_loss)
        self.log("cf_loss", cf_loss)
        return loss
# ----------------------    

    def validation_step(self, batch, _):
        x, y, c = batch
        c = c.float()
        y = y.long()
        y_hat, c_logits, _ = self.forward(x, return_concepts=True)
        return {"y": y_hat, "c": torch.sigmoid(c_logits)}

    def test_step(self, batch, _):
        x, y, c = batch
        c = c.float()
        y = y.long()
        y_hat, c_logits, _ = self.forward(x, return_concepts=True)
        return {"y": y_hat, "c": torch.sigmoid(c_logits)}

    def predict_step(self, batch, _):
        x, *_ = batch
        y_hat, c_bin = self.bi_forward(x)
        return {"y": y_hat, "c_bin": c_bin}

    def configure_optimizers(self):
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if "concept_predictor" in n],
                "lr": self.opt_cfgs.lr / 50,
                "weight_decay": 1e-2,
            },
            {
                "params": [p for n, p in self.named_parameters() if "concept_predictor" not in n],
                "weight_decay": 0,
            },
        ]
        opt = torch.optim.AdamW(param_groups, lr=self.opt_cfgs.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.opt_cfgs.max_epochs)
        return [opt], [sch]


class LinearWithRules(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.bl = torch.zeros(out_features)               # baseline logits
        self.rule2weights = []                            # 存规则-权重映射
        self.node_activation_cnt = torch.zeros(out_features)  # 激活统计

        # ✅ 关键修改：rid 对应的维度映射（字符串键）
        self.rid2dim = {f"Output_{i}": i for i in range(out_features)}

    def get_rule2weights(self, prev_layer=None, skip_from_layer=None):
        """
        返回规则和权重的对应关系
        """
        weights = self.weight.detach().cpu().numpy()
        rule2weights = []
        for i in range(self.out_features):
            node_weights = weights[i].tolist()
            rule2weights.append((f"Output_{i}", node_weights))
        self.rule2weights = rule2weights
        return self.rule2weights

    def safe_get_activation(self, rid):
        """
        ✅ 新增：安全访问 node_activation_cnt
        防止 callbacks.py 中出现 KeyError
        """
        if isinstance(rid, str):
            idx = self.rid2dim.get(rid, None)
            if idx is None:
                return 0.0  # 默认返回 0
            return self.node_activation_cnt[idx].item()
        elif isinstance(rid, int):
            return self.node_activation_cnt[rid].item()
        else:
            return 0.0



