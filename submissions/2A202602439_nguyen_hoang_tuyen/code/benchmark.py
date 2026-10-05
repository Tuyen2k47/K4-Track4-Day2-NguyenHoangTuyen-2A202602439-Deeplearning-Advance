"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo bắt buộc:
  - warmup: >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau đoạn bấm giờ
  - >= 50 lần đo (mặc định 100), báo cáo p50, p95, p99
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100,
          sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Đo thời gian chạy hàm `fn()` theo mili-giây (ms).

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize trên GPU) hoặc None trên CPU.
    """
    if sync is None and torch.cuda.is_available():
        sync = torch.cuda.synchronize

    # Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    timings = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    arr = np.array(timings, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mean = float(np.mean(arr))

    return {
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "mean": round(mean, 3),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    if device == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")
        gpu_name = "CPU (No CUDA)"
        sync_fn = None
    else:
        target_device = torch.device(device)
        gpu_name = torch.cuda.get_device_name(0) if target_device.type == "cuda" else "CPU"
        sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None

    model = model.to(target_device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device)

    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()

    @torch.inference_mode()
    def _run_forward():
        if dtype == "amp" and target_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(dummy_input)
        else:
            _ = model(dummy_input)

    res = bench(_run_forward, warmup=warmup, iters=iters, sync=sync_fn)

    images_per_s = round(batch_size / (res["p50"] / 1000.0), 1) if res["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1,
                img_size: int = 224, **kw) -> Dict[str, Any]:
    """Đo độ trễ của TTA K view (so với K lần chạy đơn lẻ)."""
    base_rep = latency_report(model, batch_size=batch_size, img_size=img_size, **kw)
    tta_rep = latency_report(model, batch_size=batch_size * k_views, img_size=img_size, **kw)
    tta_rep["k_views"] = k_views
    tta_rep["relative_cost"] = round(tta_rep["p50"] / max(base_rep["p50"], 1e-4), 2)
    return tta_rep
