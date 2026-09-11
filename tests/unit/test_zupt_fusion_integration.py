"""Unit tests for improved ZUPT integration."""

import numpy as np
import pytest

from navigation.nhc.zupt_integration import (
    ZUPTIntegrationConfig,
    ZUPTIntegrationDiagnostics,
    ZUPTIntegrationStatus,
    ZUPTIntegrator,
)


class TestZUPTIntegrator:
    """Test suite for improved ZUPT integration."""

    def test_buffer_update(self):
        """Test that buffer updates correctly."""
        config = ZUPTIntegrationConfig(window_size=5)
        integrator = ZUPTIntegrator(config)
        
        omega = np.array([0.01, 0.01, 0.01])
        force = np.array([0.0, 0.0, 9.81])
        
        # Add 5 samples
        for _ in range(5):
            integrator.update_buffer(omega, force)
        
        assert len(integrator._buffer) == 5

    def test_buffer_max_size(self):
        """Test that buffer respects max size."""
        config = ZUPTIntegrationConfig(window_size=3)
        integrator = ZUPTIntegrator(config)
        
        omega = np.array([0.01, 0.01, 0.01])
        force = np.array([0.0, 0.0, 9.81])
        
        # Add 10 samples
        for _ in range(10):
            integrator.update_buffer(omega, force)
        
        # Should only keep 3
        assert len(integrator._buffer) == 3

    def test_standstill_detection(self):
        """Test standstill detection with stationary data."""
        config = ZUPTIntegrationConfig(window_size=12)
        integrator = ZUPTIntegrator(config)
        
        # Add stationary samples (low gyro, force near g)
        for _ in range(12):
            omega = np.array([0.01, 0.01, 0.01])  # Low gyro
            force = np.array([0.0, 0.0, 9.81])  # Exactly g
            integrator.update_buffer(omega, force)
        
        is_stationary, diag = integrator.detect_standstill()
        
        assert is_stationary == True
        assert diag.status == ZUPTIntegrationStatus.APPLIED
        assert diag.applied == True
        assert diag.confidence_samples >= config.min_confidence_samples

    def test_moving_detection(self):
        """Test that moving vehicle is not detected as stationary."""
        config = ZUPTIntegrationConfig(window_size=12)
        integrator = ZUPTIntegrator(config)
        
        # Add moving samples (high gyro)
        for _ in range(12):
            omega = np.array([0.0, 0.0, 0.5])  # High yaw rate
            force = np.array([0.0, 0.0, 9.81])
            integrator.update_buffer(omega, force)
        
        is_stationary, diag = integrator.detect_standstill()
        
        assert is_stationary == False
        assert diag.status == ZUPTIntegrationStatus.SKIPPED_MOVING
        assert diag.applied == False

    def test_insufficient_buffer(self):
        """Test detection with insufficient buffer."""
        config = ZUPTIntegrationConfig(window_size=12)
        integrator = ZUPTIntegrator(config)
        
        # Add only 5 samples
        for _ in range(5):
            omega = np.array([0.01, 0.01, 0.01])
            force = np.array([0.0, 0.0, 9.81])
            integrator.update_buffer(omega, force)
        
        is_stationary, diag = integrator.detect_standstill()
        
        assert is_stationary == False
        assert diag.status == ZUPTIntegrationStatus.SKIPPED_INSUFFICIENT_DATA
        assert diag.window_size == 5

    def test_disabled_zupt(self):
        """Test that ZUPT is not applied when disabled."""
        config = ZUPTIntegrationConfig(enabled=False)
        integrator = ZUPTIntegrator(config)
        
        # Add stationary samples
        for _ in range(12):
            omega = np.array([0.01, 0.01, 0.01])
            force = np.array([0.0, 0.0, 9.81])
            integrator.update_buffer(omega, force)
        
        is_stationary, diag = integrator.detect_standstill()
        
        assert is_stationary == False
        assert diag.status == ZUPTIntegrationStatus.NOT_ATTEMPTED

    def test_partial_standstill(self):
        """Test detection with some samples meeting criteria but not enough."""
        config = ZUPTIntegrationConfig(
            window_size=12,
            min_confidence_samples=10
        )
        integrator = ZUPTIntegrator(config)
        
        # Add 8 stationary samples and 4 moving samples
        for i in range(12):
            if i < 8:
                omega = np.array([0.01, 0.01, 0.01])
                force = np.array([0.0, 0.0, 9.81])
            else:
                omega = np.array([0.0, 0.0, 0.3])
                force = np.array([0.0, 0.0, 9.81])
            integrator.update_buffer(omega, force)
        
        is_stationary, diag = integrator.detect_standstill()
        
        assert is_stationary == False
        assert diag.confidence_samples < config.min_confidence_samples

    def test_reset(self):
        """Test that reset clears the buffer."""
        config = ZUPTIntegrationConfig(window_size=5)
        integrator = ZUPTIntegrator(config)
        
        omega = np.array([0.01, 0.01, 0.01])
        force = np.array([0.0, 0.0, 9.81])
        
        for _ in range(5):
            integrator.update_buffer(omega, force)
        
        assert len(integrator._buffer) == 5
        
        integrator.reset()
        
        assert len(integrator._buffer) == 0

    def test_gyro_norm_calculation(self):
        """Test that gyro norm is calculated correctly."""
        config = ZUPTIntegrationConfig(window_size=12)
        integrator = ZUPTIntegrator(config)
        
        # Add samples with known gyro norm
        for _ in range(12):
            omega = np.array([0.03, 0.04, 0.0])  # Norm = 0.05
            force = np.array([0.0, 0.0, 9.81])
            integrator.update_buffer(omega, force)
        
        _, diag = integrator.detect_standstill()
        
        # Gyro norm should be approximately 0.05
        assert abs(diag.gyro_norm - 0.05) < 0.01

    def test_specific_force_deviation(self):
        """Test that specific force deviation is calculated correctly."""
        config = ZUPTIntegrationConfig(window_size=12)
        integrator = ZUPTIntegrator(config)
        
        # Add samples with force slightly off from g
        for _ in range(12):
            omega = np.array([0.01, 0.01, 0.01])
            force = np.array([0.0, 0.0, 9.90])  # 0.09 off from g
            integrator.update_buffer(omega, force)
        
        _, diag = integrator.detect_standstill()
        
        # Deviation should be approximately 0.09
        assert abs(diag.specific_force_deviation - 0.09) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
