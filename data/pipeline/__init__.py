"""COMPASS offline data pipeline package.

Modules:
- parse: Real IO-VNBD CSV parsing into canonical Phase 1 schemas.
- quality_tagger: Non-destructive bitmask tagging and validated stream policy.
- sync: S/V sensor and vehicle reference stream timestamp synchronization.
- outage_index: GPS outage indexing (real vs synthetic annotations).
- stationary_detect: Dual-signal low-variance stationary detector.
- manifest: Complete dataset manifest and cached array serialization.
"""

from data.pipeline.manifest import DatasetManifestBuilder, build_manifest
from data.pipeline.outage_index import OutageIndex, OutageWindow
from data.pipeline.parse import ParsedSTrip, ParsedVTrip, parse_s_file, parse_v_file
from data.pipeline.quality_tagger import QualityReport, QualityTagger, tag_imu_samples
from data.pipeline.stationary_detect import StationaryDetector, StationarySegment
from data.pipeline.sync import SyncDiagnostics, SyncMode, SyncPolicy, SyncValidationError, SynchronizedTrip, synchronize_s_v

__all__ = [
    "parse_s_file",
    "parse_v_file",
    "ParsedSTrip",
    "ParsedVTrip",
    "QualityTagger",
    "QualityReport",
    "tag_imu_samples",
    "StationaryDetector",
    "StationarySegment",
    "OutageIndex",
    "OutageWindow",
    "synchronize_s_v",
    "SynchronizedTrip",
    "SyncDiagnostics",
    "SyncMode",
    "SyncPolicy",
    "SyncValidationError",
    "DatasetManifestBuilder",
    "build_manifest",
]
