"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng `eval.compute_metrics` của repo gốc.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Nhập các module cùng thư mục
try:
    from . import dataset, model as model_utils, losses, inference
except (ImportError, ValueError):
    import dataset
    import model as model_utils
    import losses
    import inference

# Nhập eval.py từ repo gốc (nằm ở thư mục cha hoặc sys.path)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import eval as ev
except ImportError:
    # Dự phòng nếu đường dẫn chạy khác
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    params = model_utils.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(params)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> dict:
        """Lưu trọng số gốc và thay bằng trọng số shadow."""
        orig_weights = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                orig_weights[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return orig_weights

    def restore(self, model: nn.Module, orig_weights: dict) -> None:
        """Khôi phục lại trọng số gốc."""
        for name, param in model.named_parameters():
            if name in orig_weights:
                param.data.copy_(orig_weights[name])


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, scaler: Optional[torch.cuda.amp.GradScaler],
                    cfg: Config, device: torch.device, ema: Optional[EMA] = None) -> Dict[str, float]:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone bị đóng băng thì giữ BatchNorm ở eval mode
    if getattr(model, "_frozen_backbone", False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    total_loss = 0.0
    num_samples = 0
    use_cuda = (device.type == "cuda")

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        # Mixup hoặc CutMix nếu có bật
        if cfg.mix in ["mixup", "cutmix"]:
            x, targets = losses.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
            is_mixed = True
        else:
            targets = y
            is_mixed = False

        optimizer.zero_grad()

        if cfg.amp and use_cuda and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                if is_mixed:
                    loss = losses.mixed_loss(criterion, logits, targets)
                else:
                    loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            if is_mixed:
                loss = losses.mixed_loss(criterion, logits, targets)
            else:
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    avg_loss = total_loss / max(1, num_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": round(avg_loss, 4), "lr": current_lr}


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    num_samples = 0

    use_cuda = (device.type == "cuda")

    with torch.inference_mode():
        for x, y, fnames in loader:
            x = x.to(device)
            y_dev = y.to(device)
            batch_size = x.size(0)

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y_dev)
            else:
                logits = model(x)
                loss = criterion(logits, y_dev)

            total_loss += loss.item() * batch_size
            num_samples += batch_size

            all_filenames.extend(fnames)
            all_targets.append(y.numpy())
            all_logits.append(logits.float().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, num_samples)

    return all_filenames, y_true, logits, float(avg_loss)


def plot_curves(history: List[Dict[str, Any]], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h["val_top1"] for h in history]

    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1[0].plot(epochs, train_loss, label="Train Loss", marker="o", color="#1f77b4")
    ax1[0].plot(epochs, val_loss, label="Val Loss", marker="s", color="#ff7f0e")
    ax1[0].set_title(f"{title} - Loss")
    ax1[0].set_xlabel("Epoch")
    ax1[0].set_ylabel("Loss")
    ax1[0].grid(True, linestyle="--", alpha=0.5)
    ax1[0].legend()

    # Metric plot
    ax1[1].plot(epochs, val_macro_f1, label="Val Macro-F1", marker="^", color="#2ca02c")
    ax1[1].plot(epochs, val_top1, label="Val Top-1 Acc", marker="d", color="#d62728")
    ax1[1].set_title(f"{title} - Validation Metrics")
    ax1[1].set_xlabel("Epoch")
    ax1[1].set_ylabel("Score (0 - 1)")
    ax1[1].set_ylim([0.0, 1.05])
    ax1[1].grid(True, linestyle="--", alpha=0.5)
    ax1[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> Dict[str, Any]:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết."""
    # 1. Khởi tạo & thư mục lưu trữ
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Dữ liệu: load_split & check_split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, images_dir=cfg.images_dir)

    train_tfm = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tfm = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tfm,
        batch_size=cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tfm,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 3. Model
    model = model_utils.build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    )
    model.to(device)

    n_params = model_utils.count_params(model)
    gmacs = model_utils.count_gmacs(model, img_size=cfg.img_size)

    # 4. Loss
    if cfg.loss == "ce_weighted":
        c_weights = losses.class_weights(train_df["Label"].value_counts().to_dict(), beta=cfg.class_weight_beta or 0.0)
        criterion = losses.build_criterion("ce_weighted", weight=c_weights.to(device))
    elif cfg.loss == "ls":
        criterion = losses.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    elif cfg.loss == "focal":
        criterion = losses.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = losses.build_criterion("ce")

    val_criterion = nn.CrossEntropyLoss()

    # 5. Optimizer & Scheduler & Scaler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 6. Training Loop
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = r_dir / "best_model.pth"

    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá trên VAL
        if ema is not None:
            orig = ema.apply_shadow(model)
            val_fnames, val_targets, val_logits, val_loss = evaluate(model, val_loader, val_criterion, device)
            ema.restore(model, orig)
        else:
            val_fnames, val_targets, val_logits, val_loss = evaluate(model, val_loader, val_criterion, device)

        # Tính chỉ số chính thức qua eval.compute_metrics
        probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        val_metrics = ev.compute_metrics(val_targets, probs)

        val_macro_f1 = val_metrics["macro_f1"]
        val_top1 = val_metrics["top1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_top1": round(val_top1, 4),
            "time_s": round(t_epoch, 1),
            "lr": train_res["lr"],
        })

        # Lưu checkpoint tốt nhất theo MACRO-F1 VAL
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    # 7. Nạp lại checkpoint tốt nhất và xuất dự đoán
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    val_fnames, val_targets, val_logits, _ = evaluate(model, val_loader, val_criterion, device)
    val_probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
    val_probs = val_probs / np.sum(val_probs, axis=-1, keepdims=True)

    # Lưu logit val và file dự đoán val
    np.save(r_dir / "val_logits.npy", val_logits)
    ev.save_predictions(pred_path(cfg, "val"), val_fnames, val_targets, val_probs)

    # 8. Đánh giá TEST nếu được bật (Chỉ ở Bước 4)
    test_metrics = None
    if cfg.save_test_predictions:
        test_tfm = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tfm,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_fnames, test_targets, test_logits, _ = evaluate(model, test_loader, val_criterion, device)
        test_probs = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
        test_probs = test_probs / np.sum(test_probs, axis=-1, keepdims=True)

        np.save(r_dir / "test_logits.npy", test_logits)
        ev.save_predictions(pred_path(cfg, "test"), test_fnames, test_targets, test_probs)
        test_metrics = ev.compute_metrics(test_targets, test_probs)

    # 9. Lưu history và vẽ curves
    pd.DataFrame(history).to_csv(r_dir / "history.csv", index=False)
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png", title=f"{cfg.exp_id} {cfg.backbone}")

    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "params_m": n_params,
        "gmacs": gmacs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "avg_epoch_time_s": float(np.mean(epoch_times)),
        "test_metrics": test_metrics,
    }


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    cfg_fields = {f.name: f.type for f in fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Tham số không hợp lệ (thiếu '='): {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in cfg_fields:
            raise ValueError(f"Config không có thuộc tính '{k}'. Các thuộc tính: {list(cfg_fields.keys())}")

        if v.lower() in ["none", "null"]:
            out[k] = None
        elif v.lower() in ["true", "yes"]:
            out[k] = True
        elif v.lower() in ["false", "no"]:
            out[k] = False
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="+", default=[], help="Cặp KEY=VALUE để ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    res = run(cfg)
    print("Kết quả:", res)


if __name__ == "__main__":
    main()
