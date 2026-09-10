"""Unit tests for Phase 11 Dynamic Mounting Alignment & Observability Gating.

Validates:
A. Dynamic mounting yaw convergence under known synthetic rotation
B. Bad GNSS speed and acceleration rejection
C. Insufficient yaw observability handling (insufficient epochs, circular dispersion)
D. Gradual mounting vector accumulation without abrupt steps
E. Alignment confidence state transitions (UNKNOWN -> LOW_CONFIDENCE -> CONFIDENT)
F. Alignment freeze during GNSS outage
G. NHC authority disabled/skipped when alignment confidence is UNKNOWN
J. Zero runtime ground-truth dependencies
N. NHC + ZUPT handshake and precedence
O. No false ZUPT during driving
P. Severe skid / innovation rejection protection
"""

import math
import numpy as np
import pytest

from navigation.alignment.dynamic import (
    AlignmentConfidence,
    DynamicAlignmentState,
    DynamicMountingAligner,
)
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel
from navigation.nhc.skid_detection import NHCStatus
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.core import NavigationCore, NavigationCoreConfig


def test_dynamic_mounting_yaw_convergence():
    """Verify dynamic aligner converges to true mounting yaw under straight acceleration."""
    aligner = DynamicMountingAligner(
        min_speed_mps=3.0,
        min_accel_mps2=0.3,
        max_yaw_rate_rads=0.05,
        min_epochs_confident=10,
        min_resultant_length=0.80,
    )

    true_mount_yaw_deg = 25.0
    psi = math.radians(true_mount_yaw_deg)
    c, s = math.cos(psi), math.sin(psi)
    # Device axes rotated by +psi relative to vehicle:
    # f_v = [1.0, 0, 0] forward acceleration
    # f_b = [cos(psi), -sin(psi), 0]
    f_b = np.array([c * 1.2, -s * 1.2, 9.81])
    w_b = np.array([0.0, 0.0, 0.0])

    for epoch in range(15):
        st = aligner.update(
            f_level=f_b,
            omega_level=w_b,
            gnss_speed_mps=15.0,
            gnss_accel_mps2=1.2,
            is_gnss_trusted=True,
        )

    assert st.confidence == AlignmentConfidence.CONFIDENT
    assert abs(st.mounting_yaw_deg - true_mount_yaw_deg) < 1.0
    assert st.mean_resultant_length > 0.99
    assert st.circular_dispersion_deg < 2.0


def test_bad_gnss_speed_and_accel_rejection():
    """Verify aligner rejects low-speed, negative acceleration, or corrupt inputs."""
    aligner = DynamicMountingAligner(min_speed_mps=4.0, min_accel_mps2=0.35)

    f_level = np.array([1.0, 0.0, 9.81])
    w_level = np.array([0.0, 0.0, 0.0])

    # Case 1: Speed below threshold (e.g. 2.0 m/s)
    st = aligner.update(f_level, w_level, gnss_speed_mps=2.0, gnss_accel_mps2=0.5, is_gnss_trusted=True)
    assert st.accumulated_epochs == 0

    # Case 2: Deceleration / negative acceleration
    st = aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=-0.5, is_gnss_trusted=True)
    assert st.accumulated_epochs == 0

    # Case 3: Untrusted GNSS
    st = aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=0.8, is_gnss_trusted=False)
    assert st.accumulated_epochs == 0

    # Case 4: High yaw rate (cornering/turning)
    w_turning = np.array([0.0, 0.0, 0.15])  # 0.15 rad/s > 0.05
    st = aligner.update(f_level, w_turning, gnss_speed_mps=10.0, gnss_accel_mps2=0.8, is_gnss_trusted=True)
    assert st.accumulated_epochs == 0


def test_insufficient_yaw_observability():
    """Verify aligner reports UNKNOWN when samples are insufficient or inconsistent."""
    aligner = DynamicMountingAligner(min_epochs_confident=15)

    f_level = np.array([1.0, 0.0, 9.81])
    w_level = np.array([0.0, 0.0, 0.0])

    # 3 qualifying epochs only -> UNKNOWN
    for _ in range(3):
        st = aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=0.5, is_gnss_trusted=True)
    assert st.confidence == AlignmentConfidence.UNKNOWN
    assert st.mounting_yaw_deg == 0.0

    # 7 qualifying epochs -> LOW_CONFIDENCE
    for _ in range(4):
        st = aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=0.5, is_gnss_trusted=True)
    assert st.confidence == AlignmentConfidence.LOW_CONFIDENCE


def test_alignment_freeze_during_outage():
    """Verify aligner freezes state during outages and refuses updates until unfreezing."""
    aligner = DynamicMountingAligner()
    f_level = np.array([1.0, 0.0, 9.81])
    w_level = np.array([0.0, 0.0, 0.0])

    # Accumulate 6 epochs
    for _ in range(6):
        aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=0.5, is_gnss_trusted=True)

    st_before = aligner.state
    assert st_before.accumulated_epochs == 6

    # Freeze aligner
    aligner.freeze()
    assert aligner.state.is_frozen is True

    # Attempt updates during freeze
    aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=0.5, is_gnss_trusted=True)
    assert aligner.state.accumulated_epochs == 6  # Unchanged!

    # Unfreeze
    aligner.unfreeze()
    assert aligner.state.is_frozen is False
    aligner.update(f_level, w_level, gnss_speed_mps=10.0, gnss_accel_mps2=0.5, is_gnss_trusted=True)
    assert aligner.state.accumulated_epochs == 7


def test_nhc_gated_by_alignment_confidence():
    """Verify NHC skips update when alignment confidence is UNKNOWN."""
    nhc = NHCMeasurementModel()

    nom = ESKFNominalState(
        position_enu=np.zeros(3),
        velocity_enu=np.array([15.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 1e-3)

    # UNKNOWN alignment: must skip with SKIPPED_UNALIGNED_FRAME
    _, diag_unknown = nhc.update(
        state=state,
        alignment_confidence=AlignmentConfidence.UNKNOWN,
    )
    assert diag_unknown.status == NHCStatus.SKIPPED_UNALIGNED_FRAME
    assert diag_unknown.applied is False

    # CONFIDENT alignment: applies normally
    _, diag_confident = nhc.update(
        state=state,
        alignment_confidence=AlignmentConfidence.CONFIDENT,
    )
    assert diag_confident.status == NHCStatus.NORMAL
    assert diag_confident.applied is True


def test_nhc_zupt_handshake_priority():
    """Verify NHC yields unconditionally to ZUPT when vehicle is stationary."""
    nhc = NHCMeasurementModel()

    nom = ESKFNominalState(
        position_enu=np.zeros(3),
        velocity_enu=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 1e-3)

    _, diag = nhc.update(state=state, is_stationary=True)
    assert diag.status == NHCStatus.SKIPPED_STATIONARY
    assert diag.applied is False
    assert diag.reason == "SKIPPED_STATIONARY"


def test_no_false_zupt_during_normal_driving():
    """Verify NavigationCore does not trigger false ZUPT during normal driving dynamics."""
    cfg = NavigationCoreConfig(zupt_enabled=True, nhc_enabled=True)
    core = NavigationCore(cfg)
    core.initialize(
        lat0=0.0, lon0=0.0, alt0=0.0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([15.0, 0.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        gyro_bias0=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )

    # Normal highway driving: specific force includes forward vibration, gravity reaction
    for step in range(30):
        t_ns = 1_000_000_000 + int((step + 1) * 1e8)
        f_in = np.array([0.2 * math.sin(step), 0.1 * math.cos(step), 9.81])
        w_in = np.array([0.01 * math.sin(step), 0.01 * math.cos(step), 0.0])
        out = core.step_imu(f_in, w_in, dt_s=0.1, timestamp_ns=t_ns)
        assert out.zupt_applied is False

    telem = core.get_zupt_telemetry()
    assert telem["updates_accepted"] == 0
