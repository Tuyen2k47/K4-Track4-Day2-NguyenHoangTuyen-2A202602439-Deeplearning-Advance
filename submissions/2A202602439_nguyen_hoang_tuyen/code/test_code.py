"""test_code.py - kiểm tra toàn diện tính đúng đắn của code trong submission.
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import build_transforms, check_split, NUM_CLASSES
import model as model_utils
import losses
import benchmark
import inference
import train


class TestCompletedCode(unittest.TestCase):
    def test_focal_loss_gamma_zero_equals_ce(self):
        """Kiểm tra bắt buộc của RUBRIC: Focal loss với gamma=0 phải bằng đúng CrossEntropyLoss."""
        torch.manual_seed(42)
        logits = torch.randn(10, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (10,))

        ce_loss = nn.CrossEntropyLoss()(logits, targets).item()
        focal_loss = losses.FocalLoss(gamma=0.0)(logits, targets).item()

        self.assertAlmostEqual(ce_loss, focal_loss, places=5)

    def test_label_smoothing_zero_equals_ce(self):
        """Label smoothing với eps=0 phải bằng đúng CE."""
        logits = torch.randn(8, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (8,))
        ce_loss = nn.CrossEntropyLoss()(logits, targets).item()
        ls_loss = losses.LabelSmoothingCE(smoothing=0.0)(logits, targets).item()
        self.assertAlmostEqual(ce_loss, ls_loss, places=5)

    def test_class_weights(self):
        counts = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        w_inv = losses.class_weights(counts, beta=0.0)
        self.assertEqual(len(w_inv), 9)
        self.assertAlmostEqual(float(w_inv.mean().item()), 1.0, places=4)

        w_cb = losses.class_weights(counts, beta=0.99)
        self.assertEqual(len(w_cb), 9)
        self.assertAlmostEqual(float(w_cb.sum().item()), 9.0, places=4)

    def test_mixup_and_cutmix(self):
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])

        # Mixup
        x_mix, (y_a, y_b, lam) = losses.mix_batch(x, y, alpha=1.0, mode="mixup")
        self.assertEqual(x_mix.shape, x.shape)
        self.assertTrue(0.0 <= lam <= 1.0)

        # CutMix
        x_cut, (y_a, y_b, lam_cut) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
        self.assertEqual(x_cut.shape, x.shape)
        self.assertTrue(0.0 <= lam_cut <= 1.0)

    def test_freeze_backbone_and_param_groups(self):
        model = model_utils.build_model("resnet18", pretrained=False, num_classes=9, init="frozen")
        # Head phải có requires_grad == True
        classifier = model.get_classifier()
        for p in classifier.parameters():
            self.assertTrue(p.requires_grad)

        # Backbone params phải có requires_grad == False
        classifier_params = set(classifier.parameters())
        for p in model.parameters():
            if p not in classifier_params:
                self.assertFalse(p.requires_grad)

        # Kiểm tra param_groups
        groups = model_utils.param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        # Chỉ có group cho head vì backbone bị đóng băng
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["lr"], 1e-3)

    def test_temperature_scaling_and_calibration(self):
        rng = np.random.default_rng(0)
        n = 100
        # Tạo logits bị overconfident
        logits = rng.normal(size=(n, NUM_CLASSES)) * 5.0
        labels = logits.argmax(axis=1)

        t = inference.fit_temperature(logits, labels)
        self.assertGreater(t, 0.0)

        probs_cal = inference.apply_temperature(logits, t)
        np.testing.assert_allclose(probs_cal.sum(axis=-1), 1.0, atol=1e-5)

    def test_bench_and_latency(self):
        res = benchmark.bench(lambda: [i**2 for i in range(100)], warmup=5, iters=20)
        self.assertIn("p50", res)
        self.assertIn("p95", res)
        self.assertIn("p99", res)
        self.assertGreater(res["p50"], 0.0)

    def test_parse_overrides(self):
        pairs = ["seed=5", "lr_backbone=0.0002", "amp=false", "sampler=balanced", "ema_decay=none"]
        d = train.parse_overrides(pairs)
        self.assertEqual(d["seed"], 5)
        self.assertEqual(d["lr_backbone"], 0.0002)
        self.assertEqual(d["amp"], False)
        self.assertEqual(d["sampler"], "balanced")
        self.assertIsNone(d["ema_decay"])


if __name__ == "__main__":
    unittest.main()
