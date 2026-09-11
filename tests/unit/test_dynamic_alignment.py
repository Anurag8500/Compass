"""Unit tests for Dynamic Alignment and Observability Module."""

import numpy as np
import pytest

from navigation.alignment.dynamic import AlignmentConfidence, DynamicAlignment


def test_dynamic_alignment_initialization():
    """Test proper initialization and default state."""
    da = DynamicAlignment(alpha=0.1, variance_threshold=0.2, min_samples=10)
    assert da.confidence == AlignmentConfidence.UNKNOWN
    assert da.samples_processed == 0


def test_unknown_state_until_min_samples():
    """Test that confidence remains UNKNOWN until min_samples is reached."""
    da = DynamicAlignment(alpha=0.1, variance_threshold=0.1, min_samples=5)
    f_v = np.array([0.0, 1.0, 9.81])
    
    for _ in range(4):
        conf = da.update(f_v, gnss_speed=5.0)
        assert conf == AlignmentConfidence.UNKNOWN
        
    # On 5th sample, it should transition
    conf = da.update(f_v, gnss_speed=5.0)
    assert conf != AlignmentConfidence.UNKNOWN


def test_stationary_holds_confidence():
    """Test that stationary periods don't change confidence, even if var drops."""
    da = DynamicAlignment(alpha=0.5, variance_threshold=0.1, min_samples=2)
    
    # Push high variance to get high confidence
    da.update(np.array([0.0, 1.0, 9.81]), gnss_speed=5.0)
    conf = da.update(np.array([0.0, -1.0, 9.81]), gnss_speed=5.0)
    assert conf == AlignmentConfidence.HIGH_CONFIDENCE
    
    # Now simulate vehicle stopping (gnss_speed = 0.0) with no lateral variance
    for _ in range(10):
        conf = da.update(np.array([0.0, 0.0, 9.81]), gnss_speed=0.0)
        
    # Confidence should still be HIGH because we ignore zero variance while stopped
    assert conf == AlignmentConfidence.HIGH_CONFIDENCE


def test_variance_thresholds():
    """Test LOW and HIGH confidence thresholds."""
    da = DynamicAlignment(alpha=0.5, variance_threshold=1.0, min_samples=2)
    
    # Very small variance -> UNKNOWN
    da.update(np.array([0.0, 0.1, 9.81]), gnss_speed=5.0)
    conf = da.update(np.array([0.0, 0.1, 9.81]), gnss_speed=5.0)
    assert conf == AlignmentConfidence.UNKNOWN
    
    # Moderate variance -> LOW_CONFIDENCE
    da.update(np.array([0.0, 1.0, 9.81]), gnss_speed=5.0)
    conf = da.update(np.array([0.0, -1.0, 9.81]), gnss_speed=5.0)
    assert conf == AlignmentConfidence.LOW_CONFIDENCE
    
    # High variance -> HIGH_CONFIDENCE
    da.update(np.array([0.0, 4.0, 9.81]), gnss_speed=5.0)
    conf = da.update(np.array([0.0, -4.0, 9.81]), gnss_speed=5.0)
    assert conf == AlignmentConfidence.HIGH_CONFIDENCE
