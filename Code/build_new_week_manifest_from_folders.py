#!/usr/bin/env python3
"""
Build a draft manifest for new week 5/8/10 recordings from folder names.

This does not copy data. It only records absolute paths to files on the external drive.

Example:
    python Code/build_new_week_manifest_from_folders.py \
      "/Volumes/DRIVE/20240104_PD_LC_EEG_EMG_FirstBatch_Week5" \
      "/Volumes/DRIVE/20240125_PD_LC_EEG_EMG_FirstBatch_Week8" \
      "/Volumes/DRIVE/20240209_LC_PD_10wk_EEG_EMG_FirstBatch" \
      --out data/manifests/new_weeks_manifest_local.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


VIDEO_EXTS = [".avi", ".mp4", ".mov"]
RAW_EXTS = [".edf", ".mat", ".bin"]
SCORE_EXTS = [".csv", ".txt", ".mat"]


def infer_week(text: str):
    patterns = [
        r"Week\s*([0-9]+)",
        r"week\s*([0-9]+)",
        r"([0-9]+)\s*wk",
        r"([0-9]+)wk",
        r"_(?:W|w)([0-9]+)_",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def infer_mouse_id(text: str):
    m = re.search(r"Mouse\s*([0-9]+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def infer_group(text: str):
    upper = text.upper()
    if "WT" in upper:
        return "WT"
    if "PD" in upper:
        return "PD"
    return ""


def first_matching_file(folder: Path, extensions: list[str], extra_contains: list[str] | None = None):
    files = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
    files = [f for f in files if f.is_file() and not f.name.startswith("._")]

    if extra_contains:
        lower_terms = [t.lower() for t in extra_contains]
        preferred = [
            f for f in files
            if any(term in f.name.lower() for term in lower_terms)
        ]
        if preferred:
            return str(sorted(preferred)[0])

    if files:
        return str(sorted(files)[0])
    return ""


def all_matching_files(folder: Path, extensions: list[str]):
    files = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
    files = [f for f in files if f.is_file() and not f.name.startswith("._")]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path, help="Week root folders to scan.")
    parser.add_argument("--out", type=Path, default=Path("data/manifests/new_weeks_manifest_local.csv"))
    args = parser.parse_args()

    rows = []

    for root in args.roots:
        root = root.expanduser()
        if not root.exists():
            print(f"WARNING: root does not exist: {root}")
            continue

        # Treat direct subfolders containing MouseX as recording folders
        mouse_folders = [
            p for p in root.iterdir()
            if p.is_dir() and re.search(r"Mouse\s*[0-9]+", p.name, flags=re.IGNORECASE)
        ]

        # If the root itself is a mouse folder, support that too
        if re.search(r"Mouse\s*[0-9]+", root.name, flags=re.IGNORECASE):
            mouse_folders = [root]

        for folder in sorted(mouse_folders):
            combined_name = f"{root.name}_{folder.name}"

            week = infer_week(combined_name)
            mouse_id = infer_mouse_id(folder.name)
            group = infer_group(combined_name)

            raw_files = all_matching_files(folder, RAW_EXTS)
            video_files = all_matching_files(folder, VIDEO_EXTS)
            exp_files = all_matching_files(folder, [".exp"])

            raw_signal_path = str(raw_files[0]) if raw_files else ""
            video_path = str(video_files[0]) if video_files else ""
            exp_path = str(exp_files[0]) if exp_files else ""

            # Try to find scoring-like files, but avoid probability outputs if they exist
            possible_scores = all_matching_files(folder, SCORE_EXTS)
            score_candidates = [
                f for f in possible_scores
                if any(term in f.name.lower() for term in ["score", "scored", "hypno", "state", "annot"])
                and "prob" not in f.name.lower()
            ]
            manual_scores_path = str(score_candidates[0]) if score_candidates else ""

            rows.append({
                "mouse_id": mouse_id if mouse_id is not None else "",
                "group": group,
                "week": week if week is not None else "",
                "segment_id": 0,
                "recording_name": folder.name,
                "raw_signal_path": raw_signal_path,
                "manual_scores_path": manual_scores_path,
                "eegonly_prob_path": "",
                "full_prob_path": "",
                "video_path": video_path,
                "exp_path": exp_path,
                "notes": "auto-scanned; check paths manually" + ("; folder says noREM" if "norem" in folder.name.lower() else ""),
            })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote manifest: {args.out}")
    print(f"Rows: {len(df)}")
    if len(df):
        print("\nSummary:")
        print(df.groupby(["group", "week"], dropna=False).size().reset_index(name="n").to_string(index=False))
        print("\nPreview:")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
