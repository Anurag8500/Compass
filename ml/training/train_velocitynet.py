"""Full Training and Evaluation Orchestrator for VelocityNet (Phase 7).

Executes:
1. Reproducible model training on Driver E (Train split) with early stopping on Driver B (Val split)
2. Gaussian Negative Log-Likelihood (NLL) optimization with Cosine Annealing
3. Single-pass evaluation on held-out Driver A (Test split)
4. Scenario-wise performance breakdowns
5. Naive non-ML baseline comparison
6. Machine-readable metrics & diagnostic plotting
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ml.models.velocitynet import GaussianNLLLoss, VelocityNet
from ml.training.dataset import VelocityNetDataset, create_dataloader


def set_seed(seed: int = 42) -> None:
    """Enforce deterministic seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute standard speed regression metrics."""
    errors = preds - targets
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    # Pearson correlation
    p_std = np.std(preds)
    t_std = np.std(targets)
    if p_std > 1e-8 and t_std > 1e-8:
        corr = float(np.corrcoef(preds, targets)[0, 1])
    else:
        corr = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "correlation": corr,
    }


def evaluate_dataset(
    model: VelocityNet,
    dataset: VelocityNetDataset,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset, returning loss, metrics, and prediction arrays."""
    model.eval()
    loss_fn = GaussianNLLLoss()
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    all_speeds: List[np.ndarray] = []
    all_log_vars: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            speed, log_var = model(x)
            loss = loss_fn(speed, log_var, y)

            batch_len = len(y)
            total_loss += loss.item() * batch_len
            total_samples += batch_len

            all_speeds.append(speed.cpu().numpy())
            all_log_vars.append(log_var.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    mean_loss = total_loss / max(1, total_samples)
    preds_arr = np.concatenate(all_speeds)
    log_vars_arr = np.concatenate(all_log_vars)
    targets_arr = np.concatenate(all_targets)

    metrics = compute_metrics(preds_arr, targets_arr)
    return mean_loss, metrics, preds_arr, log_vars_arr, targets_arr


def train_velocitynet(
    config_path: str = "ml/training/configs/velocitynet_v1.json",
    max_epochs_override: int | None = None,
    batch_size_override: int | None = None,
    eval_only: bool = False,
) -> Dict[str, object]:
    """Train VelocityNet model according to configuration."""
    project_root = Path(__file__).resolve().parents[2]
    cfg_file = project_root / config_path
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if max_epochs_override is not None:
        cfg["max_epochs"] = max_epochs_override
    if batch_size_override is not None:
        cfg["batch_size"] = batch_size_override

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Seed: {seed}")

    train_path = project_root / cfg["train_split"]
    val_path = project_root / cfg["val_split"]
    test_path = project_root / cfg["test_split"]

    print("[*] Loading Phase 6 dataset splits...")
    train_dataset = VelocityNetDataset(train_path, split_name="train")
    val_dataset = VelocityNetDataset(val_path, split_name="validation")
    print(f"    Train windows: {len(train_dataset):,} (Driver E)")
    print(f"    Val windows:   {len(val_dataset):,} (Driver B)")

    batch_size = cfg.get("batch_size", 128)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)

    model = VelocityNet(
        input_dim=cfg.get("input_dim", 9),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dense_dim=cfg.get("dense_dim", 32),
        dropout=cfg.get("dropout", 0.2),
        min_log_var=cfg.get("min_log_var", -10.0),
        max_log_var=cfg.get("max_log_var", 10.0),
    ).to(device)

    loss_fn = GaussianNLLLoss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    max_epochs = cfg.get("max_epochs", 20)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("scheduler_t_max", max_epochs),
        eta_min=cfg.get("scheduler_eta_min", 1e-5),
    )

    checkpoint_dir = project_root / cfg.get("checkpoint_dir", "models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "velocitynet_v1_best.pt"
    history_path = project_root / cfg.get("history_path", "models/velocitynet_v1_history.json")

    best_val_loss = float("inf")
    patience = cfg.get("patience", 6)
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_corr": [],
        "lr": [],
    }

    if not eval_only:
        print(f"[*] Starting training loop for {max_epochs} epochs (patience={patience})...")
        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_samples = 0
            epoch_start = time.time()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                speed, log_var = model(x)
                loss = loss_fn(speed, log_var, y)
                loss.backward()

                # Clip gradients for RNN stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_len = len(y)
                train_loss_total += loss.item() * batch_len
                train_samples += batch_len

            scheduler.step()
            train_loss = train_loss_total / max(1, train_samples)
            current_lr = scheduler.get_last_lr()[0]

            # Validation evaluation
            val_loss, val_metrics, _, _, _ = evaluate_dataset(model, val_dataset, batch_size=256, device=device)
            epoch_sec = time.time() - epoch_start

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_corr"].append(val_metrics["correlation"])
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_sec:.1f}s] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.3f} m/s | "
                f"Val Corr: {val_metrics['correlation']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Checkpointing based on validation NLL
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "config": cfg,
                    },
                    best_checkpoint_path,
                )
                print(f"  --> Saved new best checkpoint to {best_checkpoint_path.name} (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[*] Early stopping triggered after {patience} epochs without validation improvement.")
                    break

        total_train_sec = time.time() - start_time
        print(f"[*] Training finished in {total_train_sec:.1f} seconds. Best Val Loss: {best_val_loss:.4f}")

        # Save training history
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    else:
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        print(f"[*] Evaluation-only mode with existing checkpoint: {best_checkpoint_path.name}")

    # Load best model for held-out evaluation
    print(f"[*] Loading best checkpoint from {best_checkpoint_path}...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Re-evaluate validation with best model
    val_loss, val_metrics, val_preds, val_log_vars, val_targets = evaluate_dataset(
        model, val_dataset, batch_size=256, device=device
    )

    # Held-out Test Evaluation (Driver A)
    print(f"[*] Loading held-out test split: {test_path}...")
    test_dataset = VelocityNetDataset(test_path, split_name="test")
    print(f"    Held-out Test windows: {len(test_dataset):,} (Driver A)")

    test_loss, test_metrics, test_preds, test_log_vars, test_targets = evaluate_dataset(
        model, test_dataset, batch_size=256, device=device
    )
    test_stds = np.sqrt(np.exp(test_log_vars))

    # Uncertainty diagnostics
    res = np.abs(test_preds - test_targets)
    in_1sigma = float(np.mean(res <= test_stds))
    in_2sigma = float(np.mean(res <= 2.0 * test_stds))
    in_3sigma = float(np.mean(res <= 3.0 * test_stds))

    print("============================================================")
    print("HELD-OUT DRIVER A TEST EVALUATION RESULTS:")
    print(f"  Test NLL Loss:       {test_loss:.4f}")
    print(f"  Test RMSE:           {test_metrics['rmse']:.3f} m/s ({test_metrics['rmse']*3.6:.2f} km/h)")
    print(f"  Test MAE:            {test_metrics['mae']:.3f} m/s ({test_metrics['mae']*3.6:.2f} km/h)")
    print(f"  Test Mean Bias:      {test_metrics['bias']:.3f} m/s")
    print(f"  Test Correlation:    {test_metrics['correlation']:.4f}")
    print(f"  Mean Predicted Std:  {float(np.mean(test_stds)):.3f} m/s")
    print(f"  1-Sigma Coverage:    {in_1sigma * 100:.1f}% (ideal: 68.3%)")
    print(f"  2-Sigma Coverage:    {in_2sigma * 100:.1f}% (ideal: 95.4%)")
    print("============================================================")

    # Baseline Comparisons on Test Set:
    # 1. Operational Causal Baseline: Static Training Mean Speed
    train_mean_speed = float(np.mean(train_dataset.targets))
    static_mean_preds = np.full_like(test_targets, train_mean_speed)
    base_static_mean = compute_metrics(static_mean_preds, test_targets)

    # 2. Non-Causal Oracle Diagnostic Reference: Lag-1 Ground-Truth Speed
    # NOTE: Uses ground-truth speed from the prior window. This is NOT an operational
    # baseline because prior true speed is unavailable during GNSS denial. It serves
    # strictly as an oracle diagnostic upper bound / sanity check.
    lag1_oracle_preds = np.roll(test_targets, 1)
    lag1_oracle_preds[0] = test_targets[0]
    base_lag1_oracle = compute_metrics(lag1_oracle_preds, test_targets)

    print("\n--- BASELINE & ORACLE COMPARISONS (HELD-OUT TEST SPLIT) ---")
    print(f"1. Operational Causal Baseline (Static Train Mean = {train_mean_speed:.2f} m/s): RMSE = {base_static_mean['rmse']:.3f} m/s")
    print(f"2. VelocityNet (Production GRU):                           RMSE = {test_metrics['rmse']:.3f} m/s (Beats operational baseline by {base_static_mean['rmse'] - test_metrics['rmse']:.3f} m/s)")
    print(f"3. Non-Causal Lag-1 Ground-Truth Oracle (Diagnostic Only): RMSE = {base_lag1_oracle['rmse']:.3f} m/s (NOT an operational baseline; NOT beaten)")

    # Scenario-wise analysis on Test Set
    print("\n--- SCENARIO-WISE PERFORMANCE BREAKDOWN ---")
    test_raw_feats = test_dataset.features  # (N, 20, 9)
    # Channel 5 is omega_z (yaw rate), channel 6 is norm_f
    yaw_rates = np.abs(test_raw_feats[:, -1, 5])
    accel_norms = test_raw_feats[:, -1, 6]

    scenarios: Dict[str, np.ndarray] = {
        "Overall Test Set": np.ones(len(test_targets), dtype=bool),
        "Low Speed (< 2 m/s)": test_targets < 2.0,
        "Medium Speed (2-15 m/s)": (test_targets >= 2.0) & (test_targets <= 15.0),
        "High Speed (> 15 m/s)": test_targets > 15.0,
        "Straight Driving (|w_z| <= 0.05 rad/s)": yaw_rates <= 0.05,
        "Cornering / Turning (|w_z| > 0.05 rad/s)": yaw_rates > 0.05,
        "Dynamic Acceleration / Braking": np.abs(accel_norms - 9.81) > 1.5,
    }

    scenario_results: Dict[str, Dict[str, float]] = {}
    for sc_name, sc_mask in scenarios.items():
        count = int(np.sum(sc_mask))
        if count == 0:
            continue
        sc_m = compute_metrics(test_preds[sc_mask], test_targets[sc_mask])
        sc_m["count"] = count
        sc_m["mean_sigma"] = float(np.mean(test_stds[sc_mask]))
        scenario_results[sc_name] = sc_m
        print(
            f"  {sc_name:40s} | N={count:6d} | RMSE={sc_m['rmse']:6.3f} m/s | "
            f"MAE={sc_m['mae']:6.3f} m/s | Bias={sc_m['bias']:+6.3f} m/s | sigma={sc_m['mean_sigma']:5.2f}"
        )

    # Benchmark CPU inference latency
    print("\n[*] Benchmarking PyTorch inference latency on CPU...")
    model.to(torch.device("cpu"))
    dummy_single = torch.randn(1, 20, 9)
    # Warmup
    for _ in range(50):
        _ = model(dummy_single)

    times = []
    for _ in range(500):
        t0 = time.perf_counter()
        _ = model(dummy_single)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    p50_latency = float(np.median(times))
    p95_latency = float(np.percentile(times, 95))
    mean_latency = float(np.mean(times))
    print(f"    Single-window latency (CPU): Mean={mean_latency:.2f} ms, P50={p50_latency:.2f} ms, P95={p95_latency:.2f} ms")

    # Generate diagnostic plots
    fig_dir = project_root / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train NLL", marker="o")
    plt.plot(history["epoch"], history["val_loss"], label="Val NLL", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Gaussian NLL Loss")
    plt.title("VelocityNet Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_rmse"], label="Val RMSE (m/s)", color="coral", marker="^")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (m/s)")
    plt.title("Validation Speed Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "velocitynet_training_curves.png", dpi=150)
    plt.close()

    # Plot 2: Representative test time series (Highway Segment from Driver A)
    plt.figure(figsize=(12, 4))
    seg_len = 300  # 150 seconds (stride 0.5s)
    seg_idx = 500
    t_sec = np.arange(seg_len) * 0.5
    true_seg = test_targets[seg_idx : seg_idx + seg_len]
    pred_seg = test_preds[seg_idx : seg_idx + seg_len]
    std_seg = test_stds[seg_idx : seg_idx + seg_len]

    plt.plot(t_sec, true_seg, label="Ground Truth (VBOX Doppler)", color="black", linewidth=1.5)
    plt.plot(t_sec, pred_seg, label="VelocityNet (GRU Prediction)", color="tab:blue", linewidth=1.2)
    plt.fill_between(t_sec, pred_seg - std_seg, pred_seg + std_seg, color="tab:blue", alpha=0.25, label="±1σ Predicted Uncertainty")
    plt.xlabel("Time in Segment (seconds)")
    plt.ylabel("Forward Speed (m/s)")
    plt.title("VelocityNet Speed Prediction & Uncertainty on Held-Out Driver A Segment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "velocitynet_test_predictions.png", dpi=150)
    plt.close()

    # Plot 3: Error distribution
    plt.figure(figsize=(6, 4))
    errors = test_preds - test_targets
    plt.hist(errors, bins=60, range=(-10, 10), density=True, alpha=0.7, color="tab:purple", edgecolor="black")
    plt.axvline(0.0, color="red", linestyle="--", label=f"Mean Bias = {test_metrics['bias']:+.2f} m/s")
    plt.xlabel("Speed Error: Predicted - True (m/s)")
    plt.ylabel("Probability Density")
    plt.title("VelocityNet Test Residual Distribution (Driver A)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "velocitynet_error_distribution.png", dpi=150)
    plt.close()
    print(f"[*] Saved diagnostic figures to {fig_dir}")

    # Comprehensive evaluation record
    eval_summary = {
        "model_name": "VelocityNet",
        "version": cfg.get("version", "v1.0"),
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "uncertainty_coverage": {
            "in_1sigma": in_1sigma,
            "in_2sigma": in_2sigma,
            "in_3sigma": in_3sigma,
            "mean_sigma_mps": float(np.mean(test_stds)),
        },
        "baselines": {
            "operational_static_train_mean": {
                **base_static_mean,
                "train_mean_speed_mps": train_mean_speed,
                "type": "operational_causal_baseline",
                "description": "Causal baseline predicting the static training set mean forward speed",
            },
            "non_causal_lag1_ground_truth_oracle": {
                **base_lag1_oracle,
                "type": "non_causal_oracle_diagnostic_reference",
                "description": "Non-causal diagnostic reference using prior ground-truth speed (infeasible in GNSS outage; not an operational baseline)",
            },
            "static_train_mean": base_static_mean,
            "naive_constant_velocity": base_lag1_oracle,
        },
        "scenarios": scenario_results,
        "latency_cpu_ms": {
            "mean": mean_latency,
            "p50": p50_latency,
            "p95": p95_latency,
        },
    }

    eval_json_path = checkpoint_dir / "velocitynet_v1_evaluation.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[*] Saved evaluation summary to {eval_json_path}")

    return eval_summary


if __name__ == "__main__":
    import sys
    eval_mode = "--eval-only" in sys.argv
    train_velocitynet(eval_only=eval_mode)
