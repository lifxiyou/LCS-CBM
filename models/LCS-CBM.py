# models/crl.py
# 保存版本

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import get_backbones
from models.components import (
    ConceptLayer,
    UnionLayer,
    LRLayer,
    Connection,
)


def load_balanced_loss(router_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
    """
    router_probs: [B, E] soft probs (sum=1)
    expert_mask:  [B, E] hard one-hot assignment (top-1)
    """
    num_experts = expert_mask.size(-1)
    density = torch.mean(expert_mask.float(), dim=0)          # [E]
    density_proxy = torch.mean(router_probs, dim=0)           # [E]
    loss = torch.mean(density_proxy * density) * (num_experts ** 2)
    return loss


class ConceptGate(nn.Module):
    """
    Gate takes concept_probs (sigmoid-ed) as input.
    Outputs:
      - probs: [B, E] softmax distribution
      - top1:  [B] argmax expert id
      - lb_loss: load balancing loss (soft probs + hard top1 mask)
      - entropy: mean entropy of routing distribution (for debug stats)
    """
    def __init__(self, n_concepts: int, num_experts: int, epsilon: float = 1e-8):
        super().__init__()
        self.n_concepts = n_concepts
        self.num_experts = num_experts
        self.epsilon = epsilon
        self.fc = nn.Linear(n_concepts, num_experts)

    def forward(self, concept_probs: torch.Tensor):
        logits = self.fc(concept_probs)                 # [B, E]
        probs = F.softmax(logits, dim=-1)              # [B, E], sum=1

        top1 = torch.argmax(probs, dim=-1)             # [B]
        hard_mask = F.one_hot(top1, num_classes=self.num_experts).to(probs.dtype)  # [B, E]

        lb_loss = load_balanced_loss(probs, hard_mask)
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon), dim=-1).mean()

        return probs, top1, lb_loss, entropy


class CRLExpertContinuous(nn.Module):
    """
    One expert = (ConceptLayer -> Union(l1) -> Union(l2) -> LR)
    No backbone / concept predictor inside (shared outside).
    Works with continuous concept probabilities.
    """
    def __init__(self, n_concepts: int, l1: int, l2: int, n_classes: int, use_not: bool, use_skip: bool):
        super().__init__()
        self.n_concepts = n_concepts
        self.l1 = l1
        self.l2 = l2
        self.n_classes = n_classes
        self.use_not = use_not
        self.use_skip = use_skip

        self.layer_list = nn.ModuleList([])
    

        # Build: ConceptLayer -> Union(l1) -> Union(l2) -> LR
        prev_layer_dim = n_concepts
        dims = [n_concepts, l1, l2, n_classes]

        for idx, dim in enumerate(dims):
            skip_from_layer = None

            # 只在 LR 层做 skip：把原始 concept 特征拼到 LR 输入
            if self.use_skip and idx == 3:
                # layer_list[0] 一定是 ConceptLayer
                skip_from_layer = self.layer_list[0]
                prev_layer_dim += skip_from_layer.output_dim

            if idx == 0:
                layer = ConceptLayer(dim, self.use_not)
                layer_name = "concept0"
                prev_layer_dim = layer.output_dim
            elif idx == 3:
                layer = LRLayer(prev_layer_dim, dim)
                layer_name = "lr3"
                prev_layer_dim = layer.output_dim
            else:
                # continuous=True for AND/OR
                layer_use_not = True if idx != 1 else False
                layer = UnionLayer(prev_layer_dim, dim, use_not=layer_use_not, continuous=True)
                layer_name = f"union{idx}"
                prev_layer_dim = layer.output_dim

            layer.conn = Connection(
                prev_layer=self.layer_list[-1] if len(self.layer_list) > 0 else None,
                is_skip_to_layer=False,
                skip_from_layer=skip_from_layer,
            )
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True

            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward_from_concept_probs(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """
        concept_probs: [B, C] in (0,1)
        returns class logits: [B, K]
        """
        x = concept_probs
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res

            x = layer(x)

            if layer.conn.is_skip_to_layer:
                layer.x_res = x
        return x

    @torch.no_grad()
    def binarized_forward_from_concept_probs(self, concept_probs: torch.Tensor, count: bool = False) -> torch.Tensor:
        """
        For rule extraction / counting:
          - ConceptLayer.binarized_forward uses threshold on probs
          - UnionLayer.binarized_forward uses W > 0.5
        """
        x = concept_probs
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res

            if hasattr(layer, "binarized_forward"):
                x = layer.binarized_forward(x)
            else:
                x = layer(x)

            if layer.conn.is_skip_to_layer:
                layer.x_res = x

            if count and getattr(layer, "layer_type", "") != "linear" and hasattr(layer, "node_activation_cnt"):
                if layer.node_activation_cnt is None:
                    layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=x.device)
                if layer.forward_tot is None:
                    layer.forward_tot = 0
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]

        return x

    def l2_penalty(self) -> torch.Tensor:
        # union layers + lr layer have l2_norm; concept layer doesn't
        l2 = 0.0
        for layer in self.layer_list[1:]:
            if hasattr(layer, "l2_norm"):
                l2 = l2 + layer.l2_norm()
        if not torch.is_tensor(l2):
            l2 = torch.tensor(l2, device=next(self.parameters()).device)
        return l2

    def reset_counting_stats(self, device: torch.device):
        for layer in self.layer_list[:-1]:  # exclude LR layer
            layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=device)
            layer.forward_tot = 0


class CRL(pl.LightningModule):
    """
    Multi-expert continuous CRL:
      - shared backbone + concept predictor -> concept_probs
      - gate(concept_probs) -> expert weights
      - training: mixture of expert PROBABILITIES -> y_logprob
      - ✅ add concept supervision loss (BCEWithLogits) with optional ignore -1 labels
      - predict: hard top-1 routing for counting + rule extraction
    """
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()

        self.opt_cfgs = cfgs.opt
        self.l2_weight = cfgs.model.l2_weight
        self.logic_weight = cfgs.model.get("logic_weight", 1.0)

        self.n_concepts = cfgs.data.n_concepts
        self.n_classes = cfgs.data.n_classes
        self.l1 = cfgs.model.l1
        self.l2 = cfgs.model.l2

        self.use_not = getattr(cfgs.model, "use_not", False)
        self.use_skip = getattr(cfgs.model, "use_skip", False)

        # MoE settings
        # 允许把 num_experts=0 当作“禁用 MoE/单专家基线”
        self.num_experts = max(1, int(getattr(cfgs.model, "num_experts", 4)))
        self.lb_weight = float(getattr(cfgs.model, "gate_lb_weight", 0.01))

        # temperature scaling parameter (log-space)
        self.t = nn.Parameter(torch.log(torch.tensor([cfgs.model.temperature])))

        # shared backbone + concept predictor
        self.pre_concept_model, n_features = get_backbones(cfgs.model.backbone)
        self.concept_predictor = nn.Linear(n_features, self.n_concepts)

        # gate over experts (use concept_probs)
        self.gate = ConceptGate(self.n_concepts, self.num_experts)

        # experts
        self.experts = nn.ModuleList([
            CRLExpertContinuous(
                n_concepts=self.n_concepts,
                l1=self.l1,
                l2=self.l2,
                n_classes=self.n_classes,
                use_not=self.use_not,
                use_skip=self.use_skip,
            )
            for _ in range(self.num_experts)
        ])

        # losses
        self.loss_bce = nn.BCELoss()
        self.loss_concept = nn.BCEWithLogitsLoss()

        # buffers for predict-time routing stats
        self.register_buffer("expert_usage_cnt", torch.zeros(self.num_experts, dtype=torch.long), persistent=False)
        self.top1_expert_id: int = 0  # global top-1 expert (most used during predict)

        # ---- debug gate stats on TEST ----
        self.debug_gate = bool(getattr(cfgs.model, "debug_gate", True))
        self.gate_stats_dir = getattr(cfgs.model, "gate_stats_dir", None)
        self.gate_stats_variant = getattr(cfgs.model, "gate_stats_variant", "model")
        self._gate_stats_reset()

    # ----- callback compatibility -----
    def get_top1_expert(self) -> CRLExpertContinuous:
        return self.experts[int(self.top1_expert_id)]

    @property
    def layer_list(self):
        """
        ExtractRule expects pl_module.layer_list (single chain).
        We expose the CURRENT global top-1 expert's chain here.
        """
        return self.get_top1_expert().layer_list

    # ----- helpers -----
    def l2_penalty(self):
        l2 = torch.zeros(1, device=self.device)
        for expert in self.experts:
            l2 = l2 + expert.l2_penalty()
        return l2

    def _pairwise_and_or(self, cprob):
        """
        Pairwise AND/OR directly from continuous concept probs (for logic loss).
        AND_ij = a_i * a_j
        OR_ij  = a_i + a_j - a_i*a_j
        """
        B, C = cprob.shape
        idx = torch.triu_indices(C, C, offset=1, device=cprob.device)
        ai = cprob[:, idx[0]]
        aj = cprob[:, idx[1]]
        and_val = ai * aj
        or_val = ai + aj - ai * aj
        return and_val, or_val, idx

    def _mix_expert_outputs(self, expert_logits: torch.Tensor, gate_probs: torch.Tensor):
        """
        expert_logits: [B, E, K]
        gate_probs:    [B, E] (sum=1)

        Returns:
          y_logprob: [B, K]  log of mixture probability
          mix_prob:  [B, K]
        """
        temp = torch.exp(self.t)  # scalar tensor
        expert_prob = torch.softmax(expert_logits / temp, dim=-1)           # [B,E,K]
        mix_prob = torch.sum(gate_probs.unsqueeze(-1) * expert_prob, dim=1) # [B,K]
        y_logprob = torch.log(mix_prob + 1e-8)
        return y_logprob, mix_prob

    def _concept_bce_loss(self, concept_logits: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Robust concept BCE:
        - If c contains -1 (unknown), mask them out automatically.
        - Otherwise standard BCEWithLogitsLoss.
        """
        c_float = c.float()
        if torch.any(c_float < 0):
            mask = (c_float >= 0).float()
            target = c_float.clamp(0, 1)
            per_elem = F.binary_cross_entropy_with_logits(concept_logits, target, reduction="none")
            return (per_elem * mask).sum() / (mask.sum() + 1e-8)
        else:
            return self.loss_concept(concept_logits, c_float)

    # ----- forward -----
    def forward(self, x):
        """
        Returns:
          y_logprob:      [B, K]
          concept_prob:   [B, C]
          concept_logits: [B, C]
          gate_probs:     [B, E]
          top1_idx:       [B]
          lb_loss, entropy
        """
        feats = self.pre_concept_model(x)
        concept_logits = self.concept_predictor(feats)      # [B,C]
        concept_prob = torch.sigmoid(concept_logits)        # [B,C]

        gate_probs, top1_idx, lb_loss, entropy = self.gate(concept_prob)  # [B,E], [B]

        expert_logits = torch.stack(
            [expert.forward_from_concept_probs(concept_prob) for expert in self.experts],
            dim=1
        )

        y_logprob, _ = self._mix_expert_outputs(expert_logits, gate_probs)

        return y_logprob, concept_prob, concept_logits, gate_probs, top1_idx, lb_loss, entropy

    # ----- train/val/test -----
    def training_step(self, batch, _):
        x, y, c = batch

        y_logprob, cprob, concept_logits, gate_probs, top1_idx, lb_loss, _entropy = self.forward(x)

        # classification loss in log-prob space
        cls_loss = F.nll_loss(y_logprob, y)

        # concept loss (auto-mask for -1)
        concept_loss = self._concept_bce_loss(concept_logits, c)

        # logic loss (pairwise AND/OR)
        c_float = c.float().clamp(min=0)  # if -1 exists, clamp for targets in logic loss
        and_pred, or_pred, idx = self._pairwise_and_or(cprob)
        ci = c_float[:, idx[0]]
        cj = c_float[:, idx[1]]
        and_target = ci * cj
        or_target = ci + cj - ci * cj

        and_loss = self.loss_bce(and_pred, and_target)
        or_loss = self.loss_bce(or_pred, or_target)
        logic_loss = 0.5 * (and_loss + or_loss)

        l2_loss = self.l2_weight * self.l2_penalty()
        gate_lb = self.lb_weight * lb_loss

        # 保持原有“实际生效”的行为：concept_loss 直接加到 loss
        loss = cls_loss + self.logic_weight * logic_loss + gate_lb + l2_loss + concept_loss

        self.log("cls_loss", cls_loss)
        self.log("concept_loss", concept_loss)
        self.log("and_loss", and_loss)
        self.log("or_loss", or_loss)
        self.log("logic_loss", logic_loss)
        self.log("l2_loss", l2_loss)
        self.log("gate_lb_loss", gate_lb)
        self.log("loss", loss)

        # monitor: avg gate probs + hard top1 usage
        with torch.no_grad():
            avg_probs = gate_probs.mean(dim=0)
            for i in range(self.num_experts):
                self.log(f"gate/p_expert_{i}", avg_probs[i])

            hard_cnt = torch.bincount(top1_idx, minlength=self.num_experts).float()
            hard_frac = hard_cnt / (hard_cnt.sum() + 1e-8)
            for i in range(self.num_experts):
                self.log(f"gate/hard_frac_{i}", hard_frac[i])

        return loss

    def validation_step(self, batch, _):
        x, _, _ = batch
        y_logprob, cprob, _, _, _, _, _ = self.forward(x)
        return {"y": y_logprob, "c": cprob}

    def test_step(self, batch, _):
        x, _, _ = batch
        y_logprob, cprob, _, gate_probs, top1_idx, _, entropy = self.forward(x)

        if getattr(self, "debug_gate", False):
            self._gate_stats_update(gate_probs, top1_idx, entropy)

        return {"y": y_logprob, "c": cprob}

    def on_test_epoch_start(self):
        self._gate_stats_reset()

    def on_test_epoch_end(self):
        # 避免 DDP 多卡重复打印；单卡也没影响
        if hasattr(self.trainer, "is_global_zero") and (not self.trainer.is_global_zero):
            return
        self._gate_stats_print(stage="TEST")

    # ----- predict for rule extraction -----
    def on_predict_start(self):
        device = self.device
        for expert in self.experts:
            expert.reset_counting_stats(device)
        self.expert_usage_cnt.zero_()
        self.top1_expert_id = 0

    def predict_step(self, batch, _):
        """
        Returns dict for downstream usage; ExtractRule uses counting stats on pl_module.layer_list
        (which maps to global top1 expert) after predict_end.
        """
        x, _, _ = batch

        feats = self.pre_concept_model(x)
        concept_logits = self.concept_predictor(feats)
        concept_prob = torch.sigmoid(concept_logits)

        gate_probs, top1_idx, _, _entropy = self.gate(concept_prob)

        expert_logits = torch.stack(
            [expert.forward_from_concept_probs(concept_prob) for expert in self.experts],
            dim=1
        )

        y_logprob, _ = self._mix_expert_outputs(expert_logits, gate_probs)

        # hard top1 logits + counting stats (explainability)
        B = x.shape[0]
        K = expert_logits.shape[-1]
        y_top1 = torch.empty((B, K), device=expert_logits.device, dtype=expert_logits.dtype)

        for eid, expert in enumerate(self.experts):
            mask = (top1_idx == eid)
            cnt = int(mask.sum().item())
            if cnt == 0:
                continue

            self.expert_usage_cnt[eid] += cnt
            y_e = expert.binarized_forward_from_concept_probs(concept_prob[mask], count=True)
            y_top1[mask] = y_e

        # update global top-1 expert id
        self.top1_expert_id = int(torch.argmax(self.expert_usage_cnt).item())

        return {
            "y_soft": y_logprob,
            "y_top1": y_top1,
            "gate": gate_probs,
            "top1_expert": top1_idx,
        }

    def configure_optimizers(self):
        # Keep your original grouping idea:
        # - concept predictor (and backbone) smaller lr / weight_decay
        # - others (experts + gate + temp) default lr
        param_groups = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if ("concept_predictor" in n) or ("pre_concept_model" in n)
                ],
                "lr": self.opt_cfgs.lr / 100,
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if ("concept_predictor" not in n) and ("pre_concept_model" not in n)
                ],
                "weight_decay": 0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.opt_cfgs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.opt_cfgs.max_epochs)
        return [optimizer], [scheduler]

    # ----- gate debug stats (TEST) -----
    def _gate_stats_reset(self):
        self._gate_counts = None          # [E]
        self._gate_prob_sum = None        # [E]
        self._gate_entropy_sum = 0.0
        self._gate_n = 0

    @torch.no_grad()
    def _gate_stats_update(self, gate_probs: torch.Tensor, top1_idx: torch.Tensor, entropy: torch.Tensor):
        """
        gate_probs: [B,E]
        top1_idx:   [B]
        entropy:    scalar tensor or [B] (depending on your gate implementation)
        """
        B, E = gate_probs.shape
        device = gate_probs.device

        if self._gate_counts is None:
            self._gate_counts = torch.zeros(E, device=device, dtype=torch.long)
            self._gate_prob_sum = torch.zeros(E, device=device, dtype=torch.float)

        self._gate_counts += torch.bincount(top1_idx, minlength=E)
        self._gate_prob_sum += gate_probs.sum(dim=0)

        # entropy 可能是 batch 平均的标量，也可能是 [B]
        if entropy.numel() == 1:
            self._gate_entropy_sum += float(entropy.item()) * B
        else:
            self._gate_entropy_sum += float(entropy.sum().item())

        self._gate_n += B

    def _gate_stats_summary(self, stage="TEST"):
        if self._gate_counts is None or self._gate_n == 0:
            return None

        counts = self._gate_counts.detach().cpu().float()
        prob_mean = (self._gate_prob_sum / self._gate_n).detach().cpu()
        usage_ratio = counts / counts.sum()
        ent_mean = self._gate_entropy_sum / self._gate_n
        max_usage = float(usage_ratio.max().item())

        return {
            "stage": stage,
            "variant": getattr(self, "gate_stats_variant", "model"),
            "num_experts": int(len(prob_mean)),
            "total_samples": int(counts.sum().item()),
            "top1_counts": [int(c) for c in counts.tolist()],
            "soft_assignment_mean": [float(x) for x in prob_mean.tolist()],
            "top1_usage_ratio": [float(x) for x in usage_ratio.tolist()],
            "top1_usage_pct": [float(x * 100.0) for x in usage_ratio.tolist()],
            "mean_gate_entropy": float(ent_mean),
            "max_top1_usage_ratio": max_usage,
            "max_top1_usage_pct": max_usage * 100.0,
        }

    def _gate_stats_print(self, stage="TEST"):
        if not getattr(self, "debug_gate", False):
            return

        summary = self._gate_stats_summary(stage=stage)
        if summary is None:
            print(f"[Gate stats {stage}] no data")
            return

        counts = summary["top1_counts"]
        prob_mean = summary["soft_assignment_mean"]
        usage_ratio = summary["top1_usage_ratio"]
        usage_pct = summary["top1_usage_pct"]
        ent_mean = summary["mean_gate_entropy"]
        variant = summary["variant"]

        print(f"\n[Gate stats on {stage}] variant={variant}")
        print("top1 counts:", counts, " total:", summary["total_samples"])
        print("top1 usage ratio:", [round(x, 6) for x in usage_ratio])
        print("top1 usage (%):", [round(x, 1) for x in usage_pct])
        print("soft assignment mean:", [round(x, 6) for x in prob_mean])
        print("mean gate entropy:", ent_mean)
        print("max top1 usage (%):", round(summary["max_top1_usage_pct"], 1))

        out_dir = getattr(self, "gate_stats_dir", None)
        if out_dir:
            import os
            import json
            import pandas as pd

            os.makedirs(out_dir, exist_ok=True)
            safe_variant = str(variant).replace("/", "_").replace(" ", "_")
            prefix = f"{safe_variant}_{stage.lower()}"

            rows = []
            for k in range(summary["num_experts"]):
                rows.append({
                    "variant": variant,
                    "stage": stage,
                    "subspace": k + 1,
                    "soft_assignment_mean": prob_mean[k],
                    "top1_count": counts[k],
                    "top1_usage_ratio": usage_ratio[k],
                    "top1_usage_pct": usage_pct[k],
                })
            df = pd.DataFrame(rows)
            csv_path = os.path.join(out_dir, f"subspace_usage_{prefix}.csv")
            df.to_csv(csv_path, index=False)

            summary_path = os.path.join(out_dir, f"subspace_usage_{prefix}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"saved: {csv_path}")
            print(f"saved: {summary_path}")
            self._last_gate_stats_summary = summary


