#!/usr/bin/env python3
"""
Convert EMG episode detector output into a QC-ready event table for the Streamlit app.

Example:
    python Code/make_qc_ready_emg_events.py \
      --input /path/to/EMG_episodes_NREMbaseline_on4_off2_episodegap10s.csv \
      --out-dir /path/to/qc_ready
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REM_RELEVANT = [
    "stable_REM_EMG_burst",
    "EMG_suppressed_REM",
    "mixed_REM_Wake_transition",
]


def get_col(df: pd.DataFrame, names, default=np.nan):
    for name in names:
        if name in df.columns:
            return df[name]
    return default


def make_stable_event_key(row) -> str:
    recording = str(row.get("recording_name", ""))
    group = str(row.get("group", ""))
    week = str(row.get("week", ""))
    mouse = str(row.get("mouse_id", ""))
    segment = str(row.get("segment_id", ""))
    start = float(row.get("start_sec", row.get("episode_start_sec", np.nan)))
    end = float(row.get("end_sec", row.get("episode_end_sec", np.nan)))
    return f"{recording}|{group}|W{week}|M{mouse}|seg{segment}|{start:.3f}-{end:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--out-name",
        default="EMG_episodes_NREMbaseline_qc_ready.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if len(df) == 0:
        raise ValueError(f"Input has zero rows: {args.input}")

    df = df.copy()

    if "qc_event_id" not in df.columns:
        df.insert(0, "qc_event_id", np.arange(len(df)))

    if "primary_category" not in df.columns:
        if "event_class" in df.columns:
            df["primary_category"] = df["event_class"]
        else:
            df["primary_category"] = "other_uncertain"

    if "stable_event_key" not in df.columns:
        df["stable_event_key"] = df.apply(make_stable_event_key, axis=1)

    # Probability aliases expected by the app
    df["P_REM_EEGonly"] = get_col(df, ["mean_EEGonly_P_REM", "P_REM_EEGonly"])
    df["P_Wake_EEGonly"] = get_col(df, ["mean_EEGonly_P_Awake", "mean_EEGonly_P_Wake", "P_Wake_EEGonly"])
    df["P_NREM_EEGonly"] = get_col(df, ["mean_EEGonly_P_NREM", "P_NREM_EEGonly"])

    df["P_REM_FULL"] = get_col(df, ["mean_full_P_REM", "P_REM_FULL"])
    df["P_Wake_FULL"] = get_col(df, ["mean_full_P_Awake", "mean_full_P_Wake", "P_Wake_FULL"])
    df["P_NREM_FULL"] = get_col(df, ["mean_full_P_NREM", "P_NREM_FULL"])

    if "delta_REM" not in df.columns:
        if "mean_delta_REM_EEGonly_minus_full" in df.columns:
            df["delta_REM"] = df["mean_delta_REM_EEGonly_minus_full"]
        elif "mean_delta_REM" in df.columns:
            df["delta_REM"] = df["mean_delta_REM"]
        else:
            df["delta_REM"] = df["P_REM_EEGonly"] - df["P_REM_FULL"]

    df["distance_to_transition_sec"] = get_col(
        df,
        ["min_EEGonly_distance_to_transition_sec", "distance_to_transition_sec"],
    )

    if "duration_sec_for_category" not in df.columns:
        df["duration_sec_for_category"] = get_col(df, ["duration_sec"], default=np.nan)

    if "max_EMG_z" not in df.columns:
        df["max_EMG_z"] = get_col(df, ["max_EMG_baseline_z", "max_EMG_z"], default=np.nan)

    df["rbd_priority_score"] = (
        2.0 * pd.to_numeric(df["P_REM_EEGonly"], errors="coerce").fillna(0)
        + 2.0 * pd.to_numeric(df["delta_REM"], errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(df["max_EMG_z"], errors="coerce").fillna(0)
        - 1.0 * pd.to_numeric(df["P_Wake_EEGonly"], errors="coerce").fillna(0)
    )

    df["REM_relevant_episode"] = df["primary_category"].isin(REM_RELEVANT)
    df["REM_relevant_auto"] = df["REM_relevant_episode"]

    df["phasic_episode_0p1_to_5s"] = pd.to_numeric(
        df["duration_sec_for_category"], errors="coerce"
    ).between(0.1, 5.0)

    df["long_episode_gt_5s"] = pd.to_numeric(
        df["duration_sec_for_category"], errors="coerce"
    ) > 5.0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = args.out_dir / args.out_name
    df.to_csv(out, index=False)

    criteria = pd.DataFrame([
        {
            "category": "stable_REM_EMG_burst",
            "criteria": "EEG-only P(REM) high, EEG-only P(Wake) low, far from transition",
            "interpretation": "REM-like brain state with EMG episode; strongest RBD-like category if QC confirms signal.",
        },
        {
            "category": "EMG_suppressed_REM",
            "criteria": "EEG-only P(REM) high, full-model P(REM) lower, positive delta_REM",
            "interpretation": "EEG looks REM-like, but adding EMG suppresses REM probability.",
        },
        {
            "category": "mixed_REM_Wake_transition",
            "criteria": "Intermediate REM/Wake probabilities or close to transition",
            "interpretation": "Motor activity near REM/Wake instability or transition.",
        },
        {
            "category": "wake_like_movement",
            "criteria": "EEG-only P(Wake) high and P(REM) low",
            "interpretation": "Likely ordinary wake movement.",
        },
        {
            "category": "NREM_like_EMG",
            "criteria": "EEG-only P(NREM) high",
            "interpretation": "EMG episode during NREM-like state; useful control but not primary RBD category.",
        },
        {
            "category": "other_uncertain",
            "criteria": "Does not clearly match above categories",
            "interpretation": "Needs QC or refined thresholding.",
        },
    ])

    criteria_out = args.out_dir / "NREMbaseline_episode_category_criteria.csv"
    criteria.to_csv(criteria_out, index=False)

    print("Wrote:")
    print(out)
    print(criteria_out)

    print("\nRows:", len(df))

    print("\nCategory counts:")
    print(df["primary_category"].value_counts().to_string())

    print("\nCounts by group/week:")
    print(pd.crosstab([df["group"], df["week"]], df["primary_category"]).to_string())


if __name__ == "__main__":
    main()
