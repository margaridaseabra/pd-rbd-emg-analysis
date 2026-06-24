#!/usr/bin/env python3
"""
Apply mouse_id -> genotype/group/sex mapping to manifests or event tables.

Null  -> WT
A53T  -> PD

This fixes cases where group was incorrectly inferred from folder names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MOUSE_INFO = {
    1:  {"sex": "M", "genotype": "Null", "group": "WT"},
    2:  {"sex": "F", "genotype": "Null", "group": "WT"},
    3:  {"sex": "M", "genotype": "A53T", "group": "PD"},
    4:  {"sex": "F", "genotype": "A53T", "group": "PD"},
    5:  {"sex": "M", "genotype": "A53T", "group": "PD"},
    6:  {"sex": "F", "genotype": "A53T", "group": "PD"},
    7:  {"sex": "M", "genotype": "Null", "group": "WT"},
    8:  {"sex": "F", "genotype": "Null", "group": "WT"},
    9:  {"sex": "M", "genotype": "A53T", "group": "PD"},
    10: {"sex": "F", "genotype": "Null", "group": "WT"},
    11: {"sex": "M", "genotype": "Null", "group": "WT"},
    12: {"sex": "F", "genotype": "A53T", "group": "PD"},
    13: {"sex": "M", "genotype": "A53T", "group": "PD"},
    14: {"sex": "F", "genotype": "Null", "group": "WT"},
    15: {"sex": "M", "genotype": "A53T", "group": "PD"},
}


def make_stable_event_key(row) -> str:
    recording = str(row.get("recording_name", ""))
    group = str(row.get("group", ""))
    week = str(row.get("week", ""))
    mouse = str(row.get("mouse_id", ""))
    segment = str(row.get("segment_id", ""))
    start = row.get("start_sec", row.get("episode_start_sec", np.nan))
    end = row.get("end_sec", row.get("episode_end_sec", np.nan))

    try:
        start = float(start)
        end = float(end)
        time_part = f"{start:.3f}-{end:.3f}"
    except Exception:
        time_part = "unknown_time"

    return f"{recording}|{group}|W{week}|M{mouse}|seg{segment}|{time_part}"


def patch_file(path: Path, overwrite: bool, rebuild_stable_keys: bool) -> Path:
    df = pd.read_csv(path)

    if "mouse_id" not in df.columns:
        raise ValueError(f"{path} has no mouse_id column")

    old_group = df["group"].copy() if "group" in df.columns else pd.Series([""] * len(df))

    mouse_ids = pd.to_numeric(df["mouse_id"], errors="coerce")

    unknown = sorted(set(mouse_ids.dropna().astype(int)) - set(MOUSE_INFO))
    if unknown:
        raise ValueError(f"Unknown mouse IDs in {path}: {unknown}")

    df["mouse_id"] = mouse_ids.astype("Int64")
    df["sex"] = df["mouse_id"].map(lambda x: MOUSE_INFO[int(x)]["sex"] if pd.notna(x) else "")
    df["genotype"] = df["mouse_id"].map(lambda x: MOUSE_INFO[int(x)]["genotype"] if pd.notna(x) else "")
    df["group"] = df["mouse_id"].map(lambda x: MOUSE_INFO[int(x)]["group"] if pd.notna(x) else "")

    if rebuild_stable_keys and "stable_event_key" in df.columns:
        df["stable_event_key"] = df.apply(make_stable_event_key, axis=1)

    if overwrite:
        out = path
    else:
        out = path.with_name(path.stem + "_genotype_fixed" + path.suffix)

    df.to_csv(out, index=False)

    print(f"\nPatched: {path}")
    print(f"Wrote:   {out}")
    print("Rows:", len(df))

    if "week" in df.columns:
        print("\nCounts by group/week:")
        print(pd.crosstab(df["group"], df["week"]).to_string())
    else:
        print("\nCounts by group:")
        print(df["group"].value_counts().to_string())

    changed = (old_group.astype(str).values != df["group"].astype(str).values).sum()
    print(f"\nRows whose group label changed: {changed}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-stable-keys", action="store_true")
    args = parser.parse_args()

    for path in args.csvs:
        patch_file(path, overwrite=args.overwrite, rebuild_stable_keys=args.rebuild_stable_keys)


if __name__ == "__main__":
    main()
