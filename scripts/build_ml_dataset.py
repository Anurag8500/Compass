"""CLI Script to Build and Serialize the Full Phase 6 ML Dataset.

Executes the complete Phase 6 pipeline:
1. Verifies dataset split (Driver E -> Train, Driver B -> Val, Driver A -> Test).
2. Processes all 142 downstream-ready trips through Phase 3 preprocessing,
   canonical 10 Hz resampling, 9-channel feature calculation, causal windowing,
   and VelocityNet label generation.
3. Fits FeatureNormalizer on TRAIN SPLIT ONLY.
4. Serializes train.npz, val.npz, test.npz, normalization.json, and dataset_manifest.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ml.data.builder import build_full_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and Serialize Phase 6 ML Dataset")
    parser.add_argument("--split-json", type=str, default="data/splits/split_v1.json", help="Path to split JSON")
    parser.add_argument("--output-dir", type=str, default="data/ml_dataset_v1", help="Output directory for dataset artifacts")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    manifest = build_full_dataset(
        project_root=project_root,
        split_json_path=args.split_json,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("PHASE 6 ML DATASET BUILD SUMMARY")
    print("=" * 60)
    print(f"Total Source Files:  {manifest['total_source_files_processed']}")
    print(f"Files per Split:     Train: {manifest['files_per_split']['train']}, Val: {manifest['files_per_split']['validation']}, Test: {manifest['files_per_split']['test']}")
    print(f"Drivers per Split:   Train: {manifest['drivers_per_split']['train']}, Val: {manifest['drivers_per_split']['validation']}, Test: {manifest['drivers_per_split']['test']}")
    print(f"Windows per Split:   Train: {manifest['windows_per_split']['train']}, Val: {manifest['windows_per_split']['validation']}, Test: {manifest['windows_per_split']['test']}")
    print(f"Valid Windows:       Train: {manifest['valid_windows_per_split']['train']}, Val: {manifest['valid_windows_per_split']['validation']}, Test: {manifest['valid_windows_per_split']['test']}")
    print(f"BiasNet Eligible:    Train: {manifest['biasnet_eligible_windows']['train']}, Val: {manifest['biasnet_eligible_windows']['validation']}, Test: {manifest['biasnet_eligible_windows']['test']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
