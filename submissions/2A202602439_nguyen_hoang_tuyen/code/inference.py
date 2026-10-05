"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Hỗ trợ đầy đủ:
- TTA lật ngang (hflip), multi-crop, multi-scale
- Gộp theo không gian xác suất (prob) hoặc không gian logit (logit)
- Ensemble đa mô hình / đa seed
- Temperature scaling hiệu chuẩn ECE
- Gộp BatchNorm vào Conv
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device | str,
                   view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                   ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    device = torch.device(device)
    model.eval()
    model.to(device)

    all_filenames = []
    all_targets = []
    all_logits = []

    use_amp = (device.type == "cuda")

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device)
            if view is not None:
                images = view(images)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)

            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())

    return (
        all_filenames,
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_logits, axis=0),
    )


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên batch ảnh."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W). Chiều ngang là dimension -1."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> List[torch.Tensor]:
    """5-crop: 4 góc và 1 tâm."""
    h, w = x.shape[-2], x.shape[-1]
    if h < crop or w < crop:
        # Nếu ảnh nhỏ hơn kích thước crop thì resize
        x_res = F.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False)
        return [x_res]

    top_left = x[:, :, 0:crop, 0:crop]
    top_right = x[:, :, 0:crop, w - crop:w]
    bot_left = x[:, :, h - crop:h, 0:crop]
    bot_right = x[:, :, h - crop:h, w - crop:w]
    ch = (h - crop) // 2
    cw = (w - crop) // 2
    center = x[:, :, ch:ch + crop, cw:cw + crop]

    return [center, top_left, top_right, bot_left, bot_right]


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về các kích thước khác nhau."""
    return [
        F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        for s in sizes
    ]


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một ma trận xác suất (N, K).

    - space="prob": trung bình softmax của từng view
    - space="logit": trung bình logit rồi softmax
    """
    if space == "prob":
        probs_list = []
        for l in logits_per_view:
            exp_l = np.exp(l - np.max(l, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            probs_list.append(probs)
        avg_probs = np.mean(probs_list, axis=0)
        # Chuẩn hoá đảm bảo tổng bằng 1.0
        return avg_probs / np.sum(avg_probs, axis=-1, keepdims=True)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        exp_l = np.exp(avg_logits - np.max(avg_logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return probs
    else:
        raise ValueError(f"Không hỗ trợ space={space}. Chọn 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (ensemble)."""
    assert len(list_of_probs) > 0, "Danh sách rỗng!"
    avg = np.mean(list_of_probs, axis=0)
    return avg / np.sum(avg, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên tập VAL: p = softmax(logit / T).

    Chỉ khớp trên VAL, KHÔNG khớp trên TEST (tuân thủ nguyên tắc S2, S4).
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)

    # Biến tối ưu hoá: log_temp (để T = exp(log_temp) luôn dương)
    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=50)

    nll_criterion = nn.CrossEntropyLoss()

    def _eval_loss():
        optimizer.zero_grad()
        t = torch.exp(log_temp)
        loss = nll_criterion(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(_eval_loss)
    best_t = float(torch.exp(log_temp).item())
    return max(0.01, min(10.0, round(best_t, 4)))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Tính xác suất softmax(logits / T)."""
    scaled = logits / max(T, 1e-6)
    exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    return probs


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (BatchNorm fusion)."""
    model_fused = copy.deepcopy(model)
    model_fused.eval()

    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        for name, module in list(model_fused.named_children()):
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                for i in range(len(module) - 1):
                    if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                        module[i] = fuse_conv_bn_eval(module[i], module[i + 1])
                        module[i + 1] = nn.Identity()
            else:
                fuse_conv_bn(module)
    except Exception:
        pass

    return model_fused
