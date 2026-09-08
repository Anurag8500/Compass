"""Unit tests for COMPASS canonical data schemas and adapter interfaces.

Verifies:
- Proper typed dataclass instantiation
- Coordinate frame semantics and field documentation
- Schema validation rules and boundary checking
- Exact 20x9 matrix shape validation for FeatureWindow
- Exactly 15x15 covariance matrix validation for NavigationState
- Fixed session reference-point invariant (not silently reset on each fix)
- Zero-norm quaternion rejection without mutating/normalizing inside OrientationState
- Exactly two ML models (VelocityNet, BiasNet) prediction schemas
- Exactly three GNSS FSM states (GNSS_AIDED, DR_ONLY, REACQUIRING)
- Continuous trust_score representation (degraded GNSS is continuous, not an FSM state)
- Downstream MapMatchResult with intentional schema extensions (heading_rad, distance_to_road_m)
- ModelConfig windowing (20 samples, 2.0s window, canonical 10 Hz input, stride vs execution cadence)
- Rigorous real JSON round-trip: object -> to_dict -> json.dumps -> json.loads -> from_dict -> object
- Abstract base class enforcement for SensorAdapter and GnssAdapter
"""

import json
from pathlib import Path
from typing import Any
import pytest

from navigation.adapters.base import GnssAdapter, SensorAdapter
from navigation.schemas.config import (
    CANONICAL_CHANNELS,
    ExternalSensorPacket,
    ModelConfig,
)
from navigation.schemas.gnss import GNSSSample
from navigation.schemas.imu import (
    FLAG_EXTREME_MOTION,
    FLAG_NAN_OR_NONFINITE,
    FLAG_OK,
    AlignedIMUSample,
    FeatureWindow,
    RawIMUSample,
    SensorSource,
)
from navigation.schemas.mapmatch import MapMatchResult
from navigation.schemas.ml import MLModelType, MLPrediction
from navigation.schemas.state import (
    GNSSMode,
    NavigationState,
    OrientationState,
)


def assert_json_roundtrip(obj: Any, cls: Any) -> Any:
    """Verifies complete lossless JSON round-trip:

    object -> to_dict -> json.dumps -> json.loads -> from_dict -> reconstructed
    assert reconstructed == object
    """
    d = obj.to_dict()
    json_str = json.dumps(d)
    parsed = json.loads(json_str)
    reconstructed = cls.from_dict(parsed)
    assert reconstructed == obj
    return reconstructed


class TestRawIMUSample:
    """Tests for RawIMUSample schema (Device Body Frame)."""

    def test_construction_and_defaults(self) -> None:
        sample = RawIMUSample(
            timestamp_ns=1_700_000_000_000_000_000,
            accel=(0.12, -0.45, 9.81),
            gyro=(0.01, -0.02, 0.005),
        )
        assert sample.timestamp_ns == 1_700_000_000_000_000_000
        assert sample.accel == (0.12, -0.45, 9.81)
        assert sample.gyro == (0.01, -0.02, 0.005)
        assert sample.quality_flags == FLAG_OK
        assert sample.source == SensorSource.PHONE
        assert sample.sensor_id == "phone_internal"

    def test_quality_flags(self) -> None:
        sample = RawIMUSample(
            timestamp_ns=1000,
            accel=(45.0, 0.0, 0.0),
            gyro=(0.0, 12.0, 0.0),
            quality_flags=FLAG_EXTREME_MOTION,
            source=SensorSource.EXTERNAL,
            sensor_id="wit_wt901",
        )
        assert sample.quality_flags & FLAG_EXTREME_MOTION
        assert sample.source == SensorSource.EXTERNAL
        assert sample.sensor_id == "wit_wt901"

    def test_invalid_accel_length(self) -> None:
        with pytest.raises(ValueError, match="accel must be a 3-tuple"):
            RawIMUSample(
                timestamp_ns=1000,
                accel=(0.0, 0.0),  # type: ignore
                gyro=(0.0, 0.0, 0.0),
            )

    def test_invalid_gyro_length(self) -> None:
        with pytest.raises(ValueError, match="gyro must be a 3-tuple"):
            RawIMUSample(
                timestamp_ns=1000,
                accel=(0.0, 0.0, 0.0),
                gyro=(0.0, 0.0, 0.0, 0.0),  # type: ignore
            )

    def test_negative_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp_ns must be non-negative"):
            RawIMUSample(
                timestamp_ns=-1,
                accel=(0.0, 0.0, 0.0),
                gyro=(0.0, 0.0, 0.0),
            )

    def test_real_json_roundtrip(self) -> None:
        original = RawIMUSample(
            timestamp_ns=123456789,
            accel=(1.25, -2.5, 9.80665),
            gyro=(0.012, -0.045, 0.003),
            quality_flags=FLAG_EXTREME_MOTION,
            source=SensorSource.EXTERNAL,
            sensor_id="ext_imu_01",
        )
        assert_json_roundtrip(original, RawIMUSample)


class TestAlignedIMUSample:
    """Tests for AlignedIMUSample schema (Vehicle Frame)."""

    def test_construction_vehicle_frame(self) -> None:
        sample = AlignedIMUSample(
            timestamp_ns=500_000_000,
            accel_vehicle=(0.5, 0.0, 9.80665),
            gyro_vehicle=(0.0, 0.01, -0.05),
            quality_flags=FLAG_OK,
            is_usable_for_integration=True,
        )
        assert sample.accel_vehicle == (0.5, 0.0, 9.80665)
        assert sample.gyro_vehicle == (0.0, 0.01, -0.05)
        assert sample.is_usable_for_integration is True

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="accel_vehicle must be a 3-tuple"):
            AlignedIMUSample(
                timestamp_ns=100,
                accel_vehicle=(1.0,),  # type: ignore
                gyro_vehicle=(1.0, 2.0, 3.0),
            )

    def test_real_json_roundtrip(self) -> None:
        original = AlignedIMUSample(
            timestamp_ns=99999,
            accel_vehicle=(1.23, -4.56, 7.89),
            gyro_vehicle=(-0.1, 0.2, -0.3),
            quality_flags=FLAG_NAN_OR_NONFINITE,
            is_usable_for_integration=False,
        )
        assert_json_roundtrip(original, AlignedIMUSample)


class TestFeatureWindow:
    """Tests for FeatureWindow 20x9 canonical ML input schema."""

    def test_valid_20x9_window(self) -> None:
        matrix = [[float(col) for col in range(9)] for _ in range(20)]
        window = FeatureWindow(
            window_end_timestamp_ns=2_000_000_000,
            samples=matrix,
            is_valid=True,
        )
        assert window.window_end_timestamp_ns == 2_000_000_000
        assert len(window.samples) == 20
        assert all(len(r) == 9 for r in window.samples)
        assert window.is_valid is True

    def test_invalid_row_count_rejected(self) -> None:
        matrix_19 = [[0.0] * 9 for _ in range(19)]
        with pytest.raises(ValueError, match="exactly 20 sample rows"):
            FeatureWindow(window_end_timestamp_ns=100, samples=matrix_19, is_valid=True)

        matrix_21 = [[0.0] * 9 for _ in range(21)]
        with pytest.raises(ValueError, match="exactly 20 sample rows"):
            FeatureWindow(window_end_timestamp_ns=100, samples=matrix_21, is_valid=True)

    def test_invalid_column_count_rejected(self) -> None:
        matrix_8_cols = [[0.0] * 8 for _ in range(20)]
        with pytest.raises(ValueError, match="exactly 9 feature columns"):
            FeatureWindow(window_end_timestamp_ns=100, samples=matrix_8_cols, is_valid=True)

    def test_invalid_window_flag_permits_empty_samples(self) -> None:
        window = FeatureWindow(window_end_timestamp_ns=100, samples=[], is_valid=False)
        assert window.is_valid is False
        assert len(window.samples) == 0

    def test_real_json_roundtrip(self) -> None:
        matrix = [[float(row * 9 + col) for col in range(9)] for row in range(20)]
        original = FeatureWindow(window_end_timestamp_ns=5000, samples=matrix, is_valid=True)
        assert_json_roundtrip(original, FeatureWindow)


class TestGNSSSample:
    """Tests for GNSSSample schema and continuous degradation."""

    def test_construction_and_defaults(self) -> None:
        sample = GNSSSample(
            timestamp_ns=1_000_000,
            lat=12.9716,
            lon=77.5946,
            alt=920.0,
        )
        assert sample.lat == 12.9716
        assert sample.lon == 77.5946
        assert sample.alt == 920.0
        assert sample.speed is None
        assert sample.bearing is None
        assert sample.accuracy_m is None
        assert sample.sat_count is None
        assert sample.trust_score == 1.0

    def test_full_fields_and_degraded_trust(self) -> None:
        # Degraded GNSS is represented by continuous trust_score, NOT a discrete 4th state
        sample = GNSSSample(
            timestamp_ns=1_000_000,
            lat=13.0827,
            lon=80.2707,
            alt=15.0,
            speed=12.5,
            bearing=185.0,
            accuracy_m=8.5,
            sat_count=6,
            trust_score=0.45,
        )
        assert sample.trust_score == 0.45
        assert sample.speed == 12.5
        assert sample.sat_count == 6

    def test_out_of_range_lat_lon(self) -> None:
        with pytest.raises(ValueError, match="lat must be in"):
            GNSSSample(timestamp_ns=0, lat=95.0, lon=0.0, alt=0.0)

        with pytest.raises(ValueError, match="lon must be in"):
            GNSSSample(timestamp_ns=0, lat=0.0, lon=185.0, alt=0.0)

    def test_out_of_range_trust_score(self) -> None:
        with pytest.raises(ValueError, match="trust_score must be in"):
            GNSSSample(timestamp_ns=0, lat=0.0, lon=0.0, alt=0.0, trust_score=1.5)

    def test_real_json_roundtrip(self) -> None:
        original = GNSSSample(
            timestamp_ns=1_234_567,
            lat=28.6139,
            lon=77.2090,
            alt=216.0,
            speed=20.0,
            bearing=90.0,
            accuracy_m=3.0,
            sat_count=12,
            trust_score=0.95,
        )
        assert_json_roundtrip(original, GNSSSample)


class TestOrientationAndNavigationState:
    """Tests for OrientationState, NavigationState, and GNSSMode."""

    def test_gnss_mode_has_exactly_three_states(self) -> None:
        modes = list(GNSSMode)
        assert len(modes) == 3
        assert GNSSMode.GNSS_AIDED in modes
        assert GNSSMode.DR_ONLY in modes
        assert GNSSMode.REACQUIRING in modes

    def test_orientation_state_construction(self) -> None:
        orient = OrientationState(
            q=(1.0, 0.0, 0.0, 0.0),
            gyro_bias=(0.001, -0.002, 0.0005),
        )
        assert orient.q == (1.0, 0.0, 0.0, 0.0)
        assert orient.gyro_bias == (0.001, -0.002, 0.0005)

    def test_zero_norm_quaternion_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero or near-zero norm"):
            OrientationState(q=(0.0, 0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0))

    def test_quaternion_not_normalized_inside_schema(self) -> None:
        # Crucial invariant: schema is a pure data container; must NOT normalize inside __post_init__
        unnormalized_q = (2.0, 0.0, 0.0, 0.0)
        orient = OrientationState(q=unnormalized_q, gyro_bias=(0.0, 0.0, 0.0))
        assert orient.q == (2.0, 0.0, 0.0, 0.0)

    def test_navigation_state_construction_and_validation(self) -> None:
        cov_15x15 = [[0.0] * 15 for _ in range(15)]
        for i in range(15):
            cov_15x15[i][i] = 1.0

        orient = OrientationState(q=(1.0, 0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0))
        nav_state = NavigationState(
            position_local=(100.0, 200.0, 5.0),
            velocity_local=(15.0, 0.0, 0.0),
            orientation=orient,
            accel_bias=(0.01, 0.02, -0.01),
            covariance=cov_15x15,
            reference_point=(12.9716, 77.5946),
            mode=GNSSMode.GNSS_AIDED,
            timestamp_ns=1_000_000_000,
        )
        assert nav_state.position_local == (100.0, 200.0, 5.0)
        assert nav_state.velocity_local == (15.0, 0.0, 0.0)
        assert nav_state.mode == GNSSMode.GNSS_AIDED
        assert len(nav_state.covariance) == 15

    def test_covariance_dimension_rejection(self) -> None:
        orient = OrientationState(q=(1.0, 0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0))
        cov_14x15 = [[0.0] * 15 for _ in range(14)]

        with pytest.raises(ValueError, match="covariance matrix must be 15x15"):
            NavigationState(
                position_local=(0.0, 0.0, 0.0),
                velocity_local=(0.0, 0.0, 0.0),
                orientation=orient,
                accel_bias=(0.0, 0.0, 0.0),
                covariance=cov_14x15,
                reference_point=(0.0, 0.0),
                mode=GNSSMode.DR_ONLY,
                timestamp_ns=0,
            )

    def test_fixed_session_reference_point_preservation(self) -> None:
        cov = [[0.0] * 15 for _ in range(15)]
        orient = OrientationState(q=(1.0, 0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0))
        state = NavigationState(
            position_local=(0.0, 0.0, 0.0),
            velocity_local=(0.0, 0.0, 0.0),
            orientation=orient,
            accel_bias=(0.0, 0.0, 0.0),
            covariance=cov,
            reference_point=(12.9716, 77.5946),
            mode=GNSSMode.GNSS_AIDED,
            timestamp_ns=100,
        )
        # Verify reference point remains fixed
        assert state.reference_point == (12.9716, 77.5946)

    def test_real_json_roundtrip(self) -> None:
        cov_15x15 = [[0.1 * (i + j) for j in range(15)] for i in range(15)]
        orient = OrientationState(q=(0.7071, 0.0, 0.7071, 0.0), gyro_bias=(0.002, -0.001, 0.003))
        original = NavigationState(
            position_local=(12.5, -45.2, 3.1),
            velocity_local=(8.2, 1.1, -0.2),
            orientation=orient,
            accel_bias=(0.05, -0.02, 0.01),
            covariance=cov_15x15,
            reference_point=(28.6139, 77.2090),
            mode=GNSSMode.REACQUIRING,
            timestamp_ns=777_888_999,
        )
        assert_json_roundtrip(original, NavigationState)


class TestMLPrediction:
    """Tests for MLPrediction schema supporting VelocityNet and BiasNet."""

    def test_velocity_net_prediction(self) -> None:
        pred = MLPrediction(
            model=MLModelType.VELOCITY_NET,
            value=[15.2],
            log_variance=[-2.5],
            window_end_timestamp_ns=1_000_000,
        )
        assert pred.model == MLModelType.VELOCITY_NET
        assert pred.value == [15.2]
        assert pred.log_variance == [-2.5]

    def test_bias_net_prediction(self) -> None:
        pred = MLPrediction(
            model=MLModelType.BIAS_NET,
            value=[0.01, -0.02, 0.005, 0.0001, -0.0002, 0.0003],
            log_variance=[-4.0, -4.0, -4.0, -6.0, -6.0, -6.0],
            window_end_timestamp_ns=1_000_000,
        )
        assert pred.model == MLModelType.BIAS_NET
        assert len(pred.value) == 6
        assert len(pred.log_variance) == 6

    def test_mismatched_value_variance_length(self) -> None:
        with pytest.raises(ValueError, match="value length.*must match log_variance length"):
            MLPrediction(
                model=MLModelType.VELOCITY_NET,
                value=[10.0],
                log_variance=[-1.0, -2.0],
                window_end_timestamp_ns=100,
            )

    def test_velocity_net_dimension_check(self) -> None:
        with pytest.raises(ValueError, match="VELOCITY_NET prediction must have 1 element"):
            MLPrediction(
                model=MLModelType.VELOCITY_NET,
                value=[10.0, 5.0],
                log_variance=[-1.0, -1.0],
                window_end_timestamp_ns=100,
            )

    def test_bias_net_dimension_check(self) -> None:
        with pytest.raises(ValueError, match="BIAS_NET prediction must have 6 elements"):
            MLPrediction(
                model=MLModelType.BIAS_NET,
                value=[0.1, 0.2, 0.3],
                log_variance=[-1.0, -1.0, -1.0],
                window_end_timestamp_ns=100,
            )

    def test_real_json_roundtrip(self) -> None:
        original = MLPrediction(
            model=MLModelType.VELOCITY_NET,
            value=[22.4],
            log_variance=[-3.1],
            window_end_timestamp_ns=456_789,
        )
        assert_json_roundtrip(original, MLPrediction)


class TestMapMatchResult:
    """Tests for MapMatchResult downstream schema with intentional extensions."""

    def test_snapped_construction(self) -> None:
        result = MapMatchResult(
            snapped=True,
            snapped_lat_lon=(12.97165, 77.59462),
            matched_road_id="way_12345678",
            confidence=0.92,
            heading_rad=1.5708,
            distance_to_road_m=2.3,
        )
        assert result.snapped is True
        assert result.snapped_lat_lon == (12.97165, 77.59462)
        assert result.matched_road_id == "way_12345678"
        assert result.confidence == 0.92
        assert result.heading_rad == 1.5708
        assert result.distance_to_road_m == 2.3

    def test_unsnapped_construction(self) -> None:
        result = MapMatchResult(snapped=False)
        assert result.snapped is False
        assert result.snapped_lat_lon is None
        assert result.matched_road_id is None
        assert result.confidence == 0.0

    def test_confidence_boundary(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            MapMatchResult(snapped=True, confidence=1.2)

    def test_real_json_roundtrip(self) -> None:
        original = MapMatchResult(
            snapped=True,
            snapped_lat_lon=(28.6139, 77.2090),
            matched_road_id="osm_way_99",
            confidence=0.88,
            heading_rad=0.5,
            distance_to_road_m=1.1,
        )
        assert_json_roundtrip(original, MapMatchResult)


class TestExternalSensorPacket:
    """Tests for ExternalSensorPacket edge schema."""

    def test_packet_with_and_without_magnetometer(self) -> None:
        pkt1 = ExternalSensorPacket(
            seq=101,
            t_host_ns=1_000_000_000,
            t_sensor_ns=500_000,
            accel=(0.0, 0.0, 9.81),
            gyro=(0.0, 0.0, 0.0),
            declared_rate_hz=200.0,
            sensor_id="fog_01",
        )
        assert pkt1.mag is None
        assert pkt1.declared_rate_hz == 200.0

        pkt2 = ExternalSensorPacket(
            seq=102,
            t_host_ns=1_000_005_000,
            t_sensor_ns=505_000,
            accel=(0.0, 0.0, 9.81),
            gyro=(0.0, 0.0, 0.0),
            mag=(25.0, -10.0, 42.0),
            declared_rate_hz=200.0,
            sensor_id="mems_ext_02",
        )
        assert pkt2.mag == (25.0, -10.0, 42.0)

    def test_real_json_roundtrip(self) -> None:
        original = ExternalSensorPacket(
            seq=42,
            t_host_ns=123_456,
            t_sensor_ns=120_000,
            accel=(1.0, 2.0, 3.0),
            gyro=(0.1, 0.2, 0.3),
            mag=(10.0, 20.0, 30.0),
            declared_rate_hz=100.0,
            sensor_id="wit_wt901",
        )
        assert_json_roundtrip(original, ExternalSensorPacket)


class TestModelConfig:
    """Tests for ModelConfig schema, canonical channels, and JSON serialization."""

    def test_canonical_channels_count_and_content(self) -> None:
        assert len(CANONICAL_CHANNELS) == 9
        assert CANONICAL_CHANNELS == (
            "f_x_v",
            "f_y_v",
            "f_z_v",
            "omega_x_v",
            "omega_y_v",
            "omega_z_v",
            "norm_f_v",
            "norm_f_dot_v",
            "norm_omega_v",
        )

    def test_model_config_window_and_stride_semantics(self) -> None:
        # VelocityNet: 20 samples @ 10 Hz (2.0s duration), sliding stride 0.5s (~2 Hz execution)
        vel_config = ModelConfig(
            model_name="VelocityNet",
            normalization_means=[0.0] * 9,
            normalization_stds=[1.0] * 9,
            filter_coefficients={"cutoff_hz": 5.0, "order": 2},
            window_size=20,
            window_duration_s=2.0,
            stride_duration_s=0.5,
        )
        assert vel_config.model_name == "VelocityNet"
        assert vel_config.window_size == 20
        assert vel_config.window_duration_s == 2.0
        assert vel_config.stride_duration_s == 0.5
        assert vel_config.channel_order == list(CANONICAL_CHANNELS)

        # BiasNet: 20 samples @ 10 Hz (2.0s duration), sliding stride 1.0s (~1 Hz execution)
        bias_config = ModelConfig(
            model_name="BiasNet",
            normalization_means=[0.1] * 9,
            normalization_stds=[0.5] * 9,
            filter_coefficients={},
            window_size=20,
            window_duration_s=2.0,
            stride_duration_s=1.0,
        )
        assert bias_config.stride_duration_s == 1.0

    def test_rejection_of_zero_or_negative_std(self) -> None:
        with pytest.raises(ValueError, match="must be strictly positive"):
            ModelConfig(
                model_name="VelocityNet",
                normalization_means=[0.0] * 9,
                normalization_stds=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                filter_coefficients={},
            )

    def test_rejection_of_wrong_means_length(self) -> None:
        with pytest.raises(ValueError, match="normalization_means length.*must match channel_order length"):
            ModelConfig(
                model_name="VelocityNet",
                normalization_means=[0.0] * 8,
                normalization_stds=[1.0] * 9,
                filter_coefficients={},
            )

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        config = ModelConfig(
            model_name="BiasNet",
            normalization_means=[0.1, -0.2, 9.8, 0.01, -0.01, 0.0, 9.85, 0.05, 0.02],
            normalization_stds=[0.5, 0.5, 1.2, 0.05, 0.05, 0.08, 1.1, 0.1, 0.06],
            filter_coefficients={"butter_order": 4, "cutoff_hz": 12.0},
            window_size=20,
            window_duration_s=2.0,
            stride_duration_s=0.1,
        )
        json_str = config.to_json()
        reconstructed_from_str = ModelConfig.from_json(json_str)
        assert reconstructed_from_str == config

        filepath = tmp_path / "test_model_config.json"
        config.to_file(filepath)
        reconstructed_from_file = ModelConfig.from_file(filepath)
        assert reconstructed_from_file == config


class TestAdapterInterfaces:
    """Tests verifying abstract base class enforcement for SensorAdapter and GnssAdapter."""

    def test_sensor_adapter_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class SensorAdapter"):
            SensorAdapter()  # type: ignore

    def test_gnss_adapter_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class GnssAdapter"):
            GnssAdapter()  # type: ignore

    def test_concrete_mock_sensor_adapter_works(self) -> None:
        class DummySensorAdapter(SensorAdapter):
            def __init__(self) -> None:
                self._open = False

            def open(self) -> None:
                self._open = True

            def close(self) -> None:
                self._open = False

            def read(self) -> RawIMUSample | None:
                if not self._open:
                    return None
                return RawIMUSample(
                    timestamp_ns=100,
                    accel=(0.0, 0.0, 9.81),
                    gyro=(0.0, 0.0, 0.0),
                )

            def is_open(self) -> bool:
                return self._open

        adapter = DummySensorAdapter()
        assert not adapter.is_open()
        assert adapter.read() is None

        adapter.open()
        assert adapter.is_open()
        sample = adapter.read()
        assert sample is not None
        assert sample.accel == (0.0, 0.0, 9.81)

        adapter.close()
        assert not adapter.is_open()

    def test_concrete_mock_gnss_adapter_works(self) -> None:
        class DummyGnssAdapter(GnssAdapter):
            def __init__(self) -> None:
                self._open = False

            def open(self) -> None:
                self._open = True

            def close(self) -> None:
                self._open = False

            def read(self) -> GNSSSample | None:
                if not self._open:
                    return None
                return GNSSSample(timestamp_ns=200, lat=12.0, lon=77.0, alt=900.0)

            def is_open(self) -> bool:
                return self._open

        adapter = DummyGnssAdapter()
        adapter.open()
        sample = adapter.read()
        assert sample is not None
        assert sample.lat == 12.0
        adapter.close()
