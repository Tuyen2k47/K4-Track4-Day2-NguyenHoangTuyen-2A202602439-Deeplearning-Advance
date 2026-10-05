"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hỗ trợ các họ kiến trúc timm: ResNet, ResNeXt, ConvNeXt, ViT/DeiT, Swin, EfficientNet, MobileNetV3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (pretrained and init != "scratch")
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)
        model._frozen_backbone = True
    else:
        model._frozen_backbone = False

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head phân loại.

    Backbone requires_grad = False, head phân loại requires_grad = True.
    """
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters()) if classifier is not None else set()

    for p in model.parameters():
        if p in classifier_params:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model._frozen_backbone = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:

    1. backbone weights (ndim > 1): lr = lr_backbone, weight_decay = weight_decay
    2. backbone norm & bias (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. head mới: lr = lr_head (thường gấp 10 lần), weight_decay = weight_decay
    Chỉ lấy các tham số có requires_grad == True.
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if classifier is not None else set()

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in head_params:
            head_decay.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias"):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({
            "params": backbone_decay,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        })
    if backbone_no_decay:
        groups.append({
            "params": backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0,
        })
    if head_decay:
        groups.append({
            "params": head_decay,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, img_size, img_size)
        device = next(model.parameters()).device
        dummy = dummy.to(device)
        flops = FlopCountAnalysis(model, dummy)
        # 1 MAC = 1 FLOP in fvcore convention
        gmacs = flops.total() / 1e9
        return round(float(gmacs), 2)
    except Exception:
        pass

    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size)
        device = next(model.parameters()).device
        dummy = dummy.to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return round(float(macs) / 1e9, 2)
    except Exception:
        pass

    # Bảng ước lượng tham chiếu cho các backbone tiêu chuẩn theo slide Day 2 trang 41-42
    name = getattr(model, "default_cfg", {}).get("architecture", "")
    known_macs = {
        "resnet50": 4.1,
        "resnext50_32x4d": 4.2,
        "convnext_tiny": 4.5,
        "deit_small_patch16_224": 4.6,
        "vit_small_patch16_224": 4.6,
        "swin_tiny_patch4_window7_224": 4.5,
        "efficientnet_b0": 0.39,
        "mobilenetv3_large_100": 0.22,
    }
    for k, v in known_macs.items():
        if k in name.lower():
            return v

    # Xấp xỉ theo số tham số nếu không có thư viện chuyên dụng
    params_m = count_params(model)
    return round(float(params_m * 0.16), 2)
