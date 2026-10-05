"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Cài đặt đầy đủ:
- Label smoothing CE (slide trang 56)
- Focal loss đa lớp (slide trang 57)
- Class weights (nghịch đảo & class-balanced theo số mẫu hiệu dụng)
- Mixup và CutMix (slide trang 48)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw) -> nn.Module:
    """Trả về hàm loss theo `kind`: "ce", "ls", "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loss loại: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None: đồng nhất với CrossEntropyLoss.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets]
            loss = at * loss

        return loss.mean()


def class_weights(counts: Union[list, np.ndarray, dict], beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng: w_c = (1 - beta) / (1 - beta ** n_c)
      chuẩn hoá tổng trọng số về số lớp K (Cui et al., CVPR 2019)
    """
    if isinstance(counts, dict):
        counts = [counts[k] for k in sorted(counts.keys())]
    counts_arr = np.array(counts, dtype=np.float64)
    k = len(counts_arr)

    if beta <= 0.0:
        raw_weights = 1.0 / counts_arr
        weights = raw_weights / raw_weights.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        raw_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = raw_weights / raw_weights.sum() * k

    return torch.tensor(weights, dtype=torch.float32)


def rand_bbox(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
    """Tạo toạ độ hộp chữ nhật cắt dán cho CutMix."""
    w = size[-1]
    h = size[-2]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
              mode: str = "cutmix") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix).

    Trả về (x_mixed, (y_a, y_b, lam))
    """
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        # Điều chỉnh lại lambda theo diện tích thực tế
        area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = x.size(-1) * x.size(-2)
        lam_adj = 1.0 - (area / total_area)
        return x_mixed, (y_a, y_b, lam_adj)
    else:
        return x, (y_a, y_b, 1.0)


def mixed_loss(criterion: nn.Module, logits: torch.Tensor,
               targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
