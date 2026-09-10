"""BiasNet: Deep Recurrent Neural Network for Learned IMU Bias Correction.

COMPASS Phase 8 — Learned IMU Bias Estimation.
Architecture:
- Input: (batch_size, 20, 9) causal vehicle-frame motion tensor at 10 Hz
- Recurrent Backbone: 2-layer GRU with 48 hidden units (dropout=0.1 between layers)
- Dense Intermediate: 48 -> 24 with ReLU activation
- Mean Output Head: 24 -> 6 producing:
    [delta_ba_x, delta_ba_y, delta_ba_z, delta_bg_x, delta_bg_y, delta_bg_z]
- Physical Output Clamping: Integrated into the forward computational graph:
    |delta_ba| <= bound_accel_mps2 (default 2.0 m/s^2)
    |delta_bg| <= bound_gyro_rads (default 0.15 rad/s)
- Optional Uncertainty Head (Stage B): 24 -> 6 log-variances with smooth clamping.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class BiasNet(nn.Module):
    """Production BiasNet model for learned IMU bias prediction."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dense_dim: int = 24,
        dropout: float = 0.1,
        bound_accel_mps2: float = 2.0,
        bound_gyro_rads: float = 0.15,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim = dense_dim
        self.bound_accel_mps2 = float(bound_accel_mps2)
        self.bound_gyro_rads = float(bound_gyro_rads)
        self.predict_uncertainty = predict_uncertainty

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()

        out_dim = 12 if predict_uncertainty else 6
        self.fc_out = nn.Linear(dense_dim, out_dim)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with internal physical output clamps.

        Args:
            x: (B, 20, 9) input tensor representing 2.0s history at 10 Hz.

        Returns:
            If predict_uncertainty is False:
                delta_b: (B, 6) clamped bias corrections [dba(3), dbg(3)].
            If predict_uncertainty is True:
                (delta_b, log_var): tuple of (B, 6) clamped means and (B, 6) log-variances.
        """
        # GRU outputs (output, h_n). Take final hidden state of top layer
        gru_out, _ = self.gru(x)
        # Final timestep representation
        last_hidden = gru_out[:, -1, :]

        feat = self.relu(self.fc1(last_hidden))
        raw_out = self.fc_out(feat)

        if not self.predict_uncertainty:
            # Mean head only (Stage A)
            dba_raw = raw_out[:, 0:3]
            dbg_raw = raw_out[:, 3:6]

            # In-graph hard physical clamps for numerical safety in Python/ONNX/LiteRT
            dba_clamped = torch.clamp(dba_raw, -self.bound_accel_mps2, self.bound_accel_mps2)
            dbg_clamped = torch.clamp(dbg_raw, -self.bound_gyro_rads, self.bound_gyro_rads)
            return torch.cat([dba_clamped, dbg_clamped], dim=-1)

        # Stage B: Mean and log-variance
        dba_raw = raw_out[:, 0:3]
        dbg_raw = raw_out[:, 3:6]
        log_var = raw_out[:, 6:12]

        dba_clamped = torch.clamp(dba_raw, -self.bound_accel_mps2, self.bound_accel_mps2)
        dbg_clamped = torch.clamp(dbg_raw, -self.bound_gyro_rads, self.bound_gyro_rads)
        mean_clamped = torch.cat([dba_clamped, dbg_clamped], dim=-1)
        log_var_clamped = torch.clamp(log_var, -10.0, 5.0)

        return mean_clamped, log_var_clamped
