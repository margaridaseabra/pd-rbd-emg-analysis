#!/usr/bin/env python3
"""
Fix segment_id in local Somnotate manifests using segment_XX in file paths.

Example:
    python Code/fix_segment_ids_from_paths.py data/manifests/new_weeks_somnotate_FULL_local_all_compatible.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def infer_segment_id(row) -> int | None:
    candidate_cols = [
        "file_path_preprocessed_signals",
        "file_path_raw_signals",
        "file_path_manual_state_annotation",
        "file_path_state_probabilities",
    ]

    for col in candidate_cols:
        if col not in row:
            continue
        value = str(row[col])
        m = re.search(r"segment[_-](\d+)", value)
        if m:
            return int(m.group(1))

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    new_segment_ids = []
    missing = []

    for i, row in df.iterrows():
        sid = infer_segment_id(row)
        if sid is None:
            sid = row.get("segment_id", "")
            missing.append(i)
        new_segment_ids.append(sid)

    df["segment_id"] = new_segment_ids

    out = args.out or args.manifest
    df.to_csv(out, index=False)

    print(f"Wrote: {out}")
    print(f"Rows: {len(df)}")
    print("\nsegment_id counts:")
    print(df["segment_id"].value_counts(dropna=False).sort_index().to_string())

    print("\nRows per recording/week/mouse/segment:")
    dup = (
        df.groupby(["recording_name", "group", "week", "mouse_id", "segment_id"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    bad = dup[dup["n"] > 1]
    print("Duplicate keys:", len(bad))
    if len(bad):
        print(bad.head(20).to_string(index=False))

    if missing:
        print("\nWARNING: Could not infer segment_id for rows:")
        print(missing[:20])


if __name__ == "__main__":
    main()
