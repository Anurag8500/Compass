"""Unit tests for NHC measurement model during straight driving scenarios."""

import numpy as np
import pytest

from navigation.nhc.measurement import (
    NHCConfig,
    NHCMeasurementModel,
    NHCDiagnostics,
    NHCStatus,
)
from navigation.eskf.state import ESKFNominalState, ESKFState


class TestNHCMeasurementModel:
    """Test suite for improved NHC measurement model."""

    def test_measurement_construction(self):
        """Test that NHC measurement is constructed correctly."""
        config = NHCConfig()
        model = NHCMeasurementModel(config)
        
        # Create a simple state
        nom = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([10.0, 0.0, 0.0]),  # 10 m/s forward
            q=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        P = np.eye(15) * 0.1
        state = ESKFState(nominal=nom, covariance=P)
        
        z, h_val, v_v, H, R_base, sigma_vy, sigma_vz = model.create_measurement(state)
        
        # Measurement should be zero
        assert np.allclose(z, [0.0, 0.0])
        
        # Predicted values should match transformed velocity
        # With identity quaternion, v_v = v_enu
        assert np.allclose(h_val, [0.0, 0.0])  # No lateral/vertical velocity
        
        # Jacobian should be 2x15
        assert H.shape == (2, 15)
        
        # Covariance should be diagonal
        assert np.allclose(R_base, np.diag(np.diag(R_base)))

    def test_speed_dependent_covariance(self):
        """Test that measurement covariance scales with speed."""
        config = NHCConfig()
        model = NHCMeasurementModel(config)
        
        # Low speed state
        nom_low = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([1.0, 0.0, 0.0]),  # 1 m/s
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        state_low = ESKFState(nominal=nom_low, covariance=np.eye(15) * 0.1)
        
        _, _, _, _, _, sigma_vy_low, sigma_vz_low = model.create_measurement(state_low)
        
        # High speed state
        nom_high = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([20.0, 0.0, 0.0]),  # 20 m/s
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        state_high = ESKFState(nominal=nom_high, covariance=np.eye(15) * 0.1)
        
        _, _, _, _, _, sigma_vy_high, sigma_vz_high = model.create_measurement(state_high)
        
        # High speed should have larger covariance
        assert sigma_vy_high > sigma_vy_low
        assert sigma_vz_high > sigma_vz_low

    def test_standstandstill_skip(self):
        """Test that NHC is skipped during standstill."""
        config = NHCConfig()
        model = NHCMeasurementModel(config)
        
        nom = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([0.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.1)
        
        updated_state, diag = model.update(state, is_stationary=True)
        
        assert diag.status == NHCStatus.SKIPPED_STATIONARY
        assert diag.applied == False
        assert diag.reason == "SKIPPED_STATIONARY"

    def test_low_speed_skip(self):
        """Test that NHC is skipped below minimum speed threshold."""
        config = NHCConfig(min_forward_speed_mps=0.5)
        model = NHCMeasurementModel(config)
        
        nom = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([0.3, 0.0, 0.0]),  # Below threshold
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.1)
        
        updated_state, diag = model.update(state, is_stationary=False)
        
        assert diag.status == NHCStatus.SKIPPED_LOW_SPEED
        assert diag.applied == False

    def test_normal_update_during_straight_driving(self):
        """Test normal NHC update during straight driving."""
        config = NHCConfig()
        model = NHCMeasurementModel(config)
        
        nom = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([10.0, 0.0, 0.0]),  # 10 m/s forward
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.1)
        
        updated_state, diag = model.update(state, is_stationary=False)
        
        # Should apply normally during straight driving
        assert diag.applied == True
        assert diag.status in [NHCStatus.NORMAL, NHCStatus.RELAXED]

    def test_disabled_nhc(self):
        """Test that NHC is not applied when disabled."""
        config = NHCConfig(enabled=False)
        model = NHCMeasurementModel(config)
        
        nom = ESKFNominalState.from_components(
            position_enu=np.array([0.0, 0.0, 0.0]),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=0,
        )
        state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.1)
        
        updated_state, diag = model.update(state, is_stationary=False)
        
        assert diag.status == NHCStatus.NOT_ATTEMPTED
        assert diag.applied == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
