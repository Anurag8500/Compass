"""Training pipeline for BiasNet (Phase 8).

Trains BiasNet on Driver E (Train), validates on Driver B (Validation),
and compares strictly against Zero-Correction and Training-Mean baselines.

Invariants:
- Fixed random seed for complete reproducibility.
- Early stopping on validation loss (Smooth L1).
- Hard physical bounds preserved in checkpoint.
- Driver A is STRICTLY HELD OUT.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.biasnet import BiasNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bias_split(
    npz_path: str | Path,
    filter_eligible: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load preprocessed features X and bias target labels."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["labels_constrained"]  # (M, 6) physical targets
    elig = data["is_eligible"]

    if filter_eligible:
        X = X[elig]
        y = y[elig]

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    meta = {
        "total_windows": len(elig),
        "eligible_windows": int(np.sum(elig)),
        "loaded_windows": len(X_t),
    }
    return X_t, y_t, meta


def compute_component_metrics(
    y_pred: np.ndarray, y_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute MAE, RMSE, bias, and correlation for each component."""
    dim_names = ["dba_x", "dba_y", "dba_z", "dbg_x", "dbg_y", "dbg_z"]
    results = {}

    for j in range(6):
        p = y_pred[:, j]
        t = y_true[:, j]
        diff = p - t

        mae = float(np.mean(np.abs(diff)))
        rmse = float(math.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))

        if np.std(p) > 1e-8 and np.std(t) > 1e-8:
            r = float(np.corrcoef(p, t)[0, 1])
        else:
            r = 0.0

        p50 = float(np.percentile(np.abs(diff), 50))
        p95 = float(np.percentile(np.abs(diff), 95))

        results[dim_names[j]] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "pearson_r": r,
            "abs_error_p50": p50,
            "abs_error_p95": p95,
        }

    # Vector errors
    acc_vec_err = np.linalg.norm(y_pred[:, 0:3] - y_true[:, 0:3], axis=1)
    gyr_vec_err = np.linalg.norm(y_pred[:, 3:6] - y_true[:, 3:6], axis=1)
    all_vec_err = np.linalg.norm(y_pred - y_true, axis=1)

    results["summary"] = {
        "accel_vector_rmse": float(math.sqrt(np.mean(acc_vec_err ** 2))),
        "accel_vector_mae": float(np.mean(acc_vec_err)),
        "gyro_vector_rmse": float(math.sqrt(np.mean(gyr_vec_err ** 2))),
        "gyro_vector_mae": float(np.mean(gyr_vec_err)),
        "total_vector_rmse": float(math.sqrt(np.mean(all_vec_err ** 2))),
        "total_vector_mae": float(np.mean(all_vec_err)),
    }
    return results


def train_biasnet(
    config_path: str | Path = "ml/training/configs/biasnet_v1.json",
) -> Dict[str, Any]:
    """Execute end-to-end training and baseline comparison."""
    with open(config_path) as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TrainBiasNet] Device: {device}, Config: {config_path}")

    # Load datasets
    train_path = cfg["dataset"]["train_file"]
    val_path = cfg["dataset"]["val_file"]
    filter_elig = cfg["dataset"].get("filter_eligible_only", True)

    X_train, y_train, train_meta = load_bias_split(train_path, filter_eligible=filter_elig)
    X_val, y_val, val_meta = load_bias_split(val_path, filter_eligible=filter_elig)

    print(f"  Train: {len(X_train)} windows (from {train_meta['total_windows']} total)")
    print(f"  Validation: {len(X_val)} windows (from {val_meta['total_windows']} total)")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    m_cfg = cfg["model"]
    model = BiasNet(
        input_dim=m_cfg.get("input_dim", 9),
        hidden_dim=m_cfg.get("hidden_dim", 48),
        num_layers=m_cfg.get("num_layers", 2),
        dense_dim=m_cfg.get("dense_dim", 24),
        dropout=m_cfg.get("dropout", 0.1),
        bound_accel_mps2=m_cfg.get("bound_accel_mps2", 2.0),
        bound_gyro_rads=m_cfg.get("bound_gyro_rads", 0.15),
        predict_uncertainty=m_cfg.get("predict_uncertainty", False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Initialized BiasNet ({total_params} parameters)")

    # Loss function: Smooth L1 with component weights
    beta = cfg["training"].get("beta", 0.05)
    w_acc = cfg["training"].get("accel_loss_weight", 1.0)
    w_gyr = cfg["training"].get("gyro_loss_weight", 10.0)
    smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=beta)

    weights = torch.tensor([w_acc, w_acc, w_acc, w_gyr, w_gyr, w_gyr], device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    clip_norm = cfg["training"]["gradient_clip_norm"]

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    history = {"train_loss": [], "val_loss": []}

    print(f"[TrainBiasNet] Training for up to {epochs} epochs (Patience={patience})...")
    start_time = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_train = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss_elem = smooth_l1(pred, by) * weights
            loss = loss_elem.mean()

            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            train_loss_accum += float(loss.item()) * len(bx)
            n_train += len(bx)

        train_loss_avg = train_loss_accum / max(n_train, 1)

        # Validation pass
        model.eval()
        val_loss_accum = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                pred = model(bx)
                loss_elem = smooth_l1(pred, by) * weights
                val_loss_accum += float(loss_elem.mean().item()) * len(bx)
                n_val += len(bx)

        val_loss_avg = val_loss_accum / max(n_val, 1)
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)

        is_best = val_loss_avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_avg
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = " [BEST]"
        else:
            patience_counter += 1
            flag = ""

        if ep % 5 == 0 or is_best or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}: Train Loss={train_loss_avg:.5f}, Val Loss={val_loss_avg:.5f}{flag}")

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {ep} (Patience={patience})")
            break

    total_time = time.time() - start_time
    print(f"[TrainBiasNet] Finished in {total_time:.1f}s. Best Val Loss={best_val_loss:.5f}")

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save PyTorch checkpoint
    out_models_dir = Path("models")
    out_models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_models_dir / "biasnet_v1_best.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"Saved best model checkpoint to {pt_path}")

    # Compute Final Evaluation on Driver B (Validation)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(device)).cpu().numpy()

    y_val_np = y_val.numpy()
    y_train_np = y_train.numpy()

    # Model metrics
    biasnet_metrics = compute_component_metrics(val_pred, y_val_np)

    # Baseline 1: Zero Correction (Delta_b = 0)
    zero_pred = np.zeros_like(y_val_np)
    zero_metrics = compute_component_metrics(zero_pred, y_val_np)

    # Baseline 2: Training-Set Component Mean
    train_mean = np.mean(y_train_np, axis=0)
    mean_pred = np.tile(train_mean, (len(y_val_np), 1))
    mean_metrics = compute_component_metrics(mean_pred, y_val_np)

    # Comparison summary table
    print("\n=== Direct Validation Evaluation on Driver B ===")
    print(f"{'Metric':<25} | {'Zero Baseline':<15} | {'Train Mean':<15} | {'BiasNet v1':<15}")
    print("-" * 75)
    print(f"{'Total Vector RMSE':<25} | {zero_metrics['summary']['total_vector_rmse']:<15.4f} | {mean_metrics['summary']['total_vector_rmse']:<15.4f} | {biasnet_metrics['summary']['total_vector_rmse']:<15.4f}")
    print(f"{'Accel Vector RMSE':<25} | {zero_metrics['summary']['accel_vector_rmse']:<15.4f} | {mean_metrics['summary']['accel_vector_rmse']:<15.4f} | {biasnet_metrics['summary']['accel_vector_rmse']:<15.4f}")
    print(f"{'Gyro Vector RMSE':<25} | {zero_metrics['summary']['gyro_vector_rmse']:<15.4f} | {mean_metrics['summary']['gyro_vector_rmse']:<15.4f} | {biasnet_metrics['summary']['gyro_vector_rmse']:<15.4f}")
    print("-" * 75)

    # Save model config
    config_dict = {
        "model_name": "BiasNet",
        "version": "v1.0",
        "checkpoint": str(pt_path),
        "total_parameters": total_params,
        "input_shape": [1, 20, 9],
        "output_shape": [1, 6],
        "architecture": {
            "backbone": "2-layer GRU",
            "hidden_dim": m_cfg.get("hidden_dim", 48),
            "dense_dim": m_cfg.get("dense_dim", 24),
            "dropout": m_cfg.get("dropout", 0.1),
            "bound_accel_mps2": m_cfg.get("bound_accel_mps2", 2.0),
            "bound_gyro_rads": m_cfg.get("bound_gyro_rads", 0.15),
        },
        "training_provenance": {
            "train_driver": cfg["dataset"]["train_driver"],
            "val_driver": cfg["dataset"]["val_driver"],
            "held_out_test_driver": cfg["dataset"]["test_driver"],
            "seed": cfg.get("seed", 42),
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": batch_size,
            "best_val_loss": best_val_loss,
            "training_duration_s": total_time,
        },
        "baselines_comparison_driver_b": {
            "biasnet": biasnet_metrics,
            "zero_baseline": zero_metrics,
            "train_mean_baseline": mean_metrics,
        },
    }

    cfg_out_path = out_models_dir / "model_config_biasnet_v1.json"
    with open(cfg_out_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved model config to {cfg_out_path}")

    return config_dict


if __name__ == "__main__":
    train_biasnet()
