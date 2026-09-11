"""Reproducibility and provenance metadata capture library (Phase 13).

Records:
- Git commit hash
- Software versions (Python, NumPy, PyTorch, ONNX Runtime)
- Model artifact hashes (VelocityNet, BiasNet ONNX/TorchScript)
- Map asset hash (coventry_s1_road_graph.json)
- Configuration hash
- Hardware platform & OS details
- Fixed random seeds
- Evaluation execution timestamp
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Complete provenance fingerprint for evaluation reproducibility."""
    git_commit_hash: str
    git_is_dirty: bool
    timestamp_utc: str
    python_version: str
    platform_info: str
    numpy_version: str
    random_seed: int
    config_hash: str
    map_hash: str
    velocitynet_model_hash: Optional[str] = None
    biasnet_model_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(filepath: Path) -> Optional[str]:
    """Compute SHA-256 hex digest of a file."""
    if not filepath.exists() or not filepath.is_file():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> Tuple[str, bool]:
    """Retrieve git commit hash and working tree status."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "UNKNOWN_COMMIT", False


def capture_reproducibility_metadata(
    config_dict: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    map_path: Path = Path("data/maps/coventry_s1_road_graph.json"),
    velocitynet_path: Optional[Path] = Path("models/velocitynet/velocitynet_v1.1.onnx"),
    biasnet_path: Optional[Path] = Path("models/biasnet/biasnet_v1.0.onnx"),
) -> ReproducibilityMetadata:
    """Capture a snapshot of the runtime reproducibility state."""
    commit, is_dirty = get_git_info()

    # Hash configuration
    if config_dict:
        cfg_str = json.dumps(config_dict, sort_keys=True)
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()
    else:
        cfg_hash = hashlib.sha256(b"default_config").hexdigest()

    map_hash = compute_file_sha256(map_path) or "NO_MAP"
    vn_hash = compute_file_sha256(velocitynet_path) if velocitynet_path else None
    bn_hash = compute_file_sha256(biasnet_path) if biasnet_path else None

    import numpy as np

    return ReproducibilityMetadata(
        git_commit_hash=commit,
        git_is_dirty=is_dirty,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python_version=sys.version.split()[0],
        platform_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        numpy_version=np.__version__,
        random_seed=seed,
        config_hash=cfg_hash,
        map_hash=map_hash,
        velocitynet_model_hash=vn_hash,
        biasnet_model_hash=bn_hash,
    )
