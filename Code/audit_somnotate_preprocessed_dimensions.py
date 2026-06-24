#!/usr/bin/env python3
"""
Audit Somnotate preprocessed .npy feature dimensions.

Example:
    python Code/audit_somnotate_preprocessed_dimensions.py \
      data/manifests/new_weeks_somnotate_FULL_local.csv \
      --make-subsets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--make-subsets", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    if "file_path_preprocessed_signals" not in df.columns:
        raise ValueError("Manifest must contain file_path_preprocessed_signals")

    rows = []

    for i, row in df.iterrows():
        path = Path(str(row["file_path_preprocessed_signals"]))

        if not path.exists():
            rows.append({
                "row_index": i,
                "status": "missing",
                "n_epochs": None,
                "n_features": None,
                "file_path_preprocessed_signals": str(path),
                "recording_name": row.get("recording_name", ""),
                "mouse_id": row.get("mouse_id", ""),
                "week": row.get("week", ""),
                "segment_id": row.get("segment_id", ""),
            })
            continue

        arr = np.load(path, mmap_mode="r")

        if arr.ndim != 2:
            n_epochs = arr.shape[0] if arr.ndim >= 1 else None
            n_features = None
            status = f"unexpected_ndim_{arr.ndim}"
        else:
            n_epochs, n_features = arr.shape
            status = "ok"

        rows.append({
            "row_index": i,
            "status": status,
            "n_epochs": n_epochs,
            "n_features": n_features,
            "file_path_preprocessed_signals": str(path),
            "recording_name": row.get("recording_name", ""),
            "mouse_id": row.get("mouse_id", ""),
            "week": row.get("week", ""),
            "segment_id": row.get("segment_id", ""),
            "source_exp_path": row.get("source_exp_path", ""),
        })

    audit = pd.DataFrame(rows)
    out_csv = args.manifest.with_name(args.manifest.stem + "_dimension_audit.csv")
    audit.to_csv(out_csv, index=False)

    print("\nFeature dimension summary:")
    print(audit.groupby(["status", "n_features"], dropna=False).size().reset_index(name="n").to_string(index=False))

    print("\nSummary by week/mouse/features:")
    print(
        audit.groupby(["week", "mouse_id", "n_features"], dropna=False)
        .size()
        .reset_index(name="n_segments")
        .sort_values(["n_features", "week", "mouse_id"])
        .to_string(index=False)
    )

    print(f"\nAudit written to: {out_csv}")

    if args.make_subsets:
        for n_features, sub in audit.dropna(subset=["n_features"]).groupby("n_features"):
            n_features = int(n_features)
            row_indices = sub["row_index"].astype(int).tolist()
            subset_manifest = df.iloc[row_indices].copy()

            subset_path = args.manifest.with_name(args.manifest.stem + f"_dim{n_features}.csv")
            subset_manifest.to_csv(subset_path, index=False)

            print(f"Wrote subset manifest for {n_features} features:")
            print(f"  {subset_path}")
            print(f"  rows: {len(subset_manifest)}")


if __name__ == "__main__":
    main()
