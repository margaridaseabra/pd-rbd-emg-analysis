#!/usr/bin/env python3
"""
Validate a new-weeks manifest before running the EMG/RBD pipeline.

Example:
    python Code/validate_new_week_manifest.py data/manifests/new_weeks_manifest_template.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "mouse_id",
    "group",
    "week",
    "segment_id",
    "recording_name",
    "raw_signal_path",
    "manual_scores_path",
    "eegonly_prob_path",
    "full_prob_path",
    "video_path",
    "exp_path",
    "notes",
]

ALLOWED_GROUPS = {"WT", "PD"}
EXPECTED_NEW_WEEKS = {5, 8, 10}


def path_exists_or_empty(value) -> bool:
    if pd.isna(value):
        return True
    value = str(value).strip()
    if value == "":
        return True
    return Path(value).expanduser().exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Path to new-weeks manifest CSV.")
    parser.add_argument(
        "--allow-weeks",
        nargs="*",
        type=int,
        default=sorted(EXPECTED_NEW_WEEKS),
        help="Allowed weeks. Default: 5 8 10.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    df = pd.read_csv(manifest_path)

    print(f"\nLoaded manifest: {manifest_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]

    if missing_cols:
        print("\nERROR: Missing required columns:")
        for col in missing_cols:
            print(f"  - {col}")
    else:
        print("\nRequired columns: OK")

    if extra_cols:
        print("\nExtra columns present:")
        for col in extra_cols:
            print(f"  - {col}")

    if len(df) == 0:
        print("\nManifest has zero rows. This is OK for a template.")
        return

    # Validate group
    groups = set(df["group"].dropna().astype(str).str.strip().unique())
    bad_groups = sorted(groups - ALLOWED_GROUPS)
    if bad_groups:
        print("\nERROR: Unexpected group values:")
        for g in bad_groups:
            print(f"  - {g}")
    else:
        print("\nGroup values: OK")

    # Validate week
    weeks = pd.to_numeric(df["week"], errors="coerce")
    bad_week_rows = df[weeks.isna()]
    if len(bad_week_rows):
        print("\nERROR: Rows with invalid week values:")
        print(bad_week_rows[["mouse_id", "group", "week", "recording_name"]])

    allowed_weeks = set(args.allow_weeks)
    present_weeks = set(weeks.dropna().astype(int).unique())
    unexpected_weeks = sorted(present_weeks - allowed_weeks)
    if unexpected_weeks:
        print("\nWARNING: Unexpected weeks found:")
        for w in unexpected_weeks:
            print(f"  - {w}")
        print(f"Allowed weeks were: {sorted(allowed_weeks)}")
    else:
        print("\nWeek values: OK")

    # Required path columns for processing
    path_columns = [
        "raw_signal_path",
        "manual_scores_path",
        "eegonly_prob_path",
        "full_prob_path",
        "video_path",
        "exp_path",
    ]

    print("\nPath check:")
    for col in path_columns:
        if col not in df.columns:
            continue
        nonempty = df[col].notna() & (df[col].astype(str).str.strip() != "")
        missing = df[nonempty & ~df[col].apply(path_exists_or_empty)]
        print(f"  {col}: {nonempty.sum()} filled, {len(missing)} missing/not found")
        if len(missing):
            print(missing[["mouse_id", "group", "week", "recording_name", col]].head(10))

    print("\nSummary by group/week:")
    summary = (
        df.assign(week_num=weeks)
        .groupby(["group", "week_num"], dropna=False)
        .size()
        .reset_index(name="n_recordings")
    )
    print(summary.to_string(index=False))

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
