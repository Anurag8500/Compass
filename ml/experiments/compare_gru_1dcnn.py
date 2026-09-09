"""Comparative Benchmark Experiment: GRU (VelocityNet) vs 1D-CNN Baseline.

As specified in FINAL_IMPLEMENTATION_PLAN_SIH26168.md (Phase 7 Task 7):
- Train/evaluate lightweight 1D Temporal Convolutional network experimental baseline.
- Evaluate speed RMSE, MAE, parameter count, and inference latency.
- The GRU remains the authoritative production VelocityNet architecture specified by COMPASS;
  the 1D-CNN is evaluated solely as an experimental lightweight alternative. The reported subset
  experiment did not establish universal superiority of the GRU.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from ml.models.baselines.cnn1d_velocity import CNN1DVelocityBaseline
from ml.models.velocitynet import GaussianNLLLoss, VelocityNet
from ml.training.dataset import VelocityNetDataset, create_dataloader
from ml.training.train_velocitynet import compute_metrics, set_seed


def run_comparison(
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
) -> Dict[str, object]:
    """Run comparative benchmark between GRU and 1D-CNN."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]

    val_path = project_root / "data" / "ml_dataset_v1" / "validation.npz"
    test_path = project_root / "data" / "ml_dataset_v1" / "test.npz"
    train_path = project_root / "data" / "ml_dataset_v1" / "train.npz"

    print("[*] Loading datasets for comparison experiment...")
    train_ds = VelocityNetDataset(train_path, split_name="train")
    val_ds = VelocityNetDataset(val_path, split_name="validation")
    test_ds = VelocityNetDataset(test_path, split_name="test")

    # Use a representative subset for the comparative experiment (e.g. 50,000 train samples)
    # to evaluate architecture training dynamics cleanly
    subset_size = min(50000, len(train_ds))
    indices = np.random.RandomState(seed).permutation(len(train_ds))[:subset_size]
    sub_feat = train_ds.features[indices]
    sub_targ = train_ds.targets[indices]

    sub_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(sub_feat), torch.from_numpy(sub_targ)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(val_ds, batch_size=256, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=256, shuffle=False)

    loss_fn = GaussianNLLLoss()

    models = {
        "VelocityNet (2L-GRU)": VelocityNet(input_dim=9, hidden_dim=64, num_layers=2, dense_dim=32).to(device),
        "Baseline (1D-CNN)": CNN1DVelocityBaseline(input_dim=9, channels=(48, 64, 64), kernel_size=3, dense_dim=32).to(device),
    }

    results: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n==========================================")
        print(f"Training {name} ({param_count:,} parameters) on {subset_size:,} samples...")
        print(f"==========================================")

        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        train_start = time.time()

        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            n_samples = 0
            for bx, by in sub_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                sp, lv = model(bx)
                loss = loss_fn(sp, lv, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                ep_loss += loss.item() * len(by)
                n_samples += len(by)
            print(f"  Epoch {ep}/{epochs} | Train Loss: {ep_loss/n_samples:.4f}")

        train_time = time.time() - train_start

        # Evaluate on Test Split
        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                sp, _ = model(bx)
                test_preds.append(sp.cpu().numpy())
                test_targets.append(by.numpy())

        test_preds_arr = np.concatenate(test_preds)
        test_targets_arr = np.concatenate(test_targets)
        test_metrics = compute_metrics(test_preds_arr, test_targets_arr)

        # Measure CPU inference latency
        model.to(torch.device("cpu"))
        dummy = torch.randn(1, 20, 9)
        for _ in range(30):
            _ = model(dummy)
        latencies = []
        for _ in range(300):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        results[name] = {
            "parameters": param_count,
            "train_time_sec": train_time,
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_corr": test_metrics["correlation"],
            "latency_p50_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
        }
        print(f"  --> {name} Test RMSE: {test_metrics['rmse']:.3f} m/s | Latency (P50): {results[name]['latency_p50_ms']:.2f} ms")

    out_path = project_root / "models" / "gru_vs_1dcnn_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Saved comparison results to {out_path}")
    return results


if __name__ == "__main__":
    run_comparison(epochs=4)
