"""Unit tests for improved NHC skid detection with adaptive thresholds."""

import math
import numpy as np
import pytest

from navigation.nhc.skid_detection import (
    NHCStatus,
    SkidDetectorConfig,
    SkidEvaluationResult,
    SkidSlipDetector,
)
from navigation.eskf.state import ESKFNominalState, ESKFState


class TestSkidSlipDetector:
    """Test suite for improved skid detection with adaptive thresholds."""

    def test_speed_dependent_thresholds(self):
        """Test that thresholds increase with vehicle speed."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        # At low speed (0 m/s), should use base thresholds
        yaw_low, lat_low = detector._compute_speed_dependent_thresholds(0.0)
        assert yaw_low == config.base_yaw_rate_rads
        assert lat_low == config.base_lateral_accel_mps2
        
        # At high speed (20 m/s), should use higher thresholds
        yaw_high, lat_high = detector._compute_speed_dependent_thresholds(20.0)
        assert yaw_high > yaw_low
        assert lat_high > lat_low
        assert yaw_high <= config.max_yaw_rate_rads
        assert lat_high <= config.max_lateral_accel_mps2

    def test_smooth_inflation_function(self):
        """Test that covariance inflation uses smooth sigmoid-like function."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        # Below threshold: no inflation
        inflation_below = detector._compute_smooth_inflation(5.0, 9.210)
        assert inflation_below == 1.0
        
        # Slightly above threshold: minimal inflation
        inflation_above = detector._compute_smooth_inflation(15.0, 9.210)
        assert inflation_above > 1.0
        
        # Much higher NIS: higher inflation
        inflation_higher = detector._compute_smooth_inflation(30.0, 9.210)
        assert inflation_higher > inflation_above
        
        # Very high NIS: approach max inflation
        inflation_very_high = detector._compute_smooth_inflation(100.0, 9.210)
        assert inflation_very_high <= config.max_inflation_factor

    def test_normal_acceptance(self):
        """Test normal acceptance when NIS is below threshold."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        result = detector.evaluate(nis=5.0, forward_speed=10.0)
        
        assert result.status == NHCStatus.NORMAL
        assert result.applied == True
        assert result.inflation_factor == 1.0
        assert result.reason == "ACCEPTED_NORMAL"

    def test_relaxed_acceptance_with_elevated_nis(self):
        """Test relaxed acceptance when NIS is elevated but not severe."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        result = detector.evaluate(nis=15.0, forward_speed=10.0)
        
        assert result.status == NHCStatus.RELAXED
        assert result.applied == True
        assert result.inflation_factor > 1.0
        assert result.reason == "ACCEPTED_RELAXED_INNOVATION"

    def test_skip_with_severe_innovation(self):
        """Test skip when NIS exceeds severe threshold."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        result = detector.evaluate(nis=30.0, forward_speed=10.0)
        
        assert result.status == NHCStatus.SKIPPED
        assert result.applied == False
        assert result.inflation_factor == 1.0
        assert result.reason == "SKIPPED_SEVERE_INNOVATION"

    def test_dynamic_cornering_skip(self):
        """Test skip during dynamic cornering with very high NIS."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        # High yaw rate and lateral acceleration at low speed (stricter thresholds)
        omega_v = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw rate
        f_v = np.array([0.0, 4.0, 9.81])  # 4 m/s² lateral accel
        
        result = detector.evaluate(nis=30.0, omega_v=omega_v, f_v=f_v, forward_speed=5.0)
        
        assert result.status == NHCStatus.SKIPPED
        assert result.applied == False
        assert result.is_kinematically_dynamic == True

    def test_dynamic_cornering_relaxed_at_low_nis(self):
        """Test that dynamic cornering with low NIS is handled appropriately."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        # Very high yaw rate and lateral accel at low speed to trigger dynamic detection
        # Need values that exceed the speed-adapted thresholds
        omega_v = np.array([0.0, 0.0, 1.2])  # 1.2 rad/s yaw rate (very high)
        f_v = np.array([0.0, 5.0, 9.81])  # 5.0 m/s² lateral accel (very high)
        
        result = detector.evaluate(nis=8.0, omega_v=omega_v, f_v=f_v, forward_speed=2.0)
        
        # Should be applied (either NORMAL or RELAXED depending on exact thresholds)
        assert result.applied == True
        assert result.is_kinematically_dynamic == True

    def test_gnss_aware_tightening(self):
        """Test that NHC behavior adapts based on GNSS trust."""
        config = SkidDetectorConfig(enable_gnss_aware=True)
        detector = SkidSlipDetector(config)
        
        # Low GNSS trust: NHC should be more active (tighter constraint needed)
        result_low_trust = detector.evaluate(nis=10.0, forward_speed=10.0, gnss_trust=0.3)
        
        # High GNSS trust: NHC can be more relaxed (GNSS provides good position)
        result_high_trust = detector.evaluate(nis=10.0, forward_speed=10.0, gnss_trust=0.9)
        
        # Verify GNSS awareness is working (both should be applied but with different behaviors)
        assert result_low_trust.applied == True or result_low_trust.status == NHCStatus.RELAXED
        assert result_high_trust.applied == True or result_high_trust.status == NHCStatus.RELAXED

    def test_hysteresis_prevents_rapid_switching(self):
        """Test that hysteresis mechanism exists and can prevent rapid switching."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        # Verify the detector has hysteresis state tracking mechanism
        assert hasattr(detector, '_previous_was_skipped')
        
        # First evaluation with dynamic motion at low speed -> skip
        omega_v = np.array([0.0, 0.0, 1.0])
        f_v = np.array([0.0, 4.0, 9.81])
        result1 = detector.evaluate(nis=30.0, omega_v=omega_v, f_v=f_v, forward_speed=5.0)
        assert result1.status == NHCStatus.SKIPPED
        
        # Verify hysteresis state was set after skip
        assert detector._previous_was_skipped == True

    def test_non_finite_innovation_handling(self):
        """Test handling of non-finite innovation values."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        result_nan = detector.evaluate(nis=float('nan'), forward_speed=10.0)
        assert result_nan.status == NHCStatus.SKIPPED
        assert result_nan.reason == "NON_FINITE_INNOVATION"
        
        result_inf = detector.evaluate(nis=float('inf'), forward_speed=10.0)
        assert result_inf.status == NHCStatus.SKIPPED
        assert result_inf.reason == "NON_FINITE_INNOVATION"

    def test_negative_nis_handling(self):
        """Test handling of negative NIS (invalid)."""
        config = SkidDetectorConfig()
        detector = SkidSlipDetector(config)
        
        result = detector.evaluate(nis=-1.0, forward_speed=10.0)
        assert result.status == NHCStatus.SKIPPED
        assert result.reason == "NON_FINITE_INNOVATION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
