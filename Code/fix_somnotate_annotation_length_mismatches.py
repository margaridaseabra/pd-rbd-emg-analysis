#!/usr/bin/env python3
"""
Fix small length mismatches between Somnotate preprocessed arrays and manual annotation TSV files.

This edits only the prepared Somnotate annotation files, not the original manual scoring files.

Example dry run:
    python Code/fix_somnotate_annotation_length_mismatches.py data/manifests/new_weeks_somnotate_FULL_local.csv

Apply fixes:
    python Code/fix_somnotate_annotation_length_mismatches.py data/manifests/new_weeks_somnotate_FULL_local.csv --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_stage_duration_tsv(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()

    header_lines = []
    stage_rows = []

    for line in lines:
        if not line.strip():
            continue
        if line.startswith("*"):
            header_lines.append(line)
        else:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            state = parts[0].strip()
            end_time = float(parts[1])
            stage_rows.append((state, end_time))

    if not stage_rows:
        raise ValueError(f"No stage rows found in {path}")

    return header_lines, stage_rows


def rewrite_stage_duration_tsv(path: Path, header_lines, stage_rows, target_duration_s: float):
    new_header = []
    has_duration = False

    for line in header_lines:
        if line.startswith("*Duration_sec"):
            new_header.append(f"*Duration_sec\t{target_duration_s:.6f}")
            has_duration = True
        else:
            new_header.append(line)

    if not has_duration:
        new_header.insert(0, f"*Duration_sec\t{target_duration_s:.6f}")

    out_lines = list(new_header)
    for state, end_time in stage_rows:
        out_lines.append(f"{state}\t{end_time:.6f}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def adjust_stage_rows(stage_rows, target_duration_s: float):
    current_duration_s = stage_rows[-1][1]

    # Case 1: annotation is shorter than signal: extend final state.
    if target_duration_s > current_duration_s:
        new_rows = list(stage_rows)
        last_state, _ = new_rows[-1]
        new_rows[-1] = (last_state, target_duration_s)
        return new_rows

    # Case 2: annotation is longer than signal: trim to target duration.
    new_rows = []
    state_at_target = stage_rows[-1][0]

    previous_end = 0.0
    for state, end_time in stage_rows:
        if target_duration_s <= end_time:
            state_at_target = state
            break
        new_rows.append((state, end_time))
        previous_end = end_time

    if new_rows and new_rows[-1][0] == state_at_target:
        new_rows[-1] = (state_at_target, target_duration_s)
    else:
        new_rows.append((state_at_target, target_duration_s))

    return new_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--epoch-sec", type=float, default=5.0)
    parser.add_argument("--max-fix-epochs", type=int, default=10)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    required = ["file_path_preprocessed_signals", "file_path_manual_state_annotation"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    rows = []

    for i, row in df.iterrows():
        prep_path = Path(str(row["file_path_preprocessed_signals"]))
        ann_path = Path(str(row["file_path_manual_state_annotation"]))

        if not prep_path.exists():
            rows.append({"row": i, "status": "missing_preprocessed", "path": str(prep_path)})
            continue

        if not ann_path.exists():
            rows.append({"row": i, "status": "missing_annotation", "path": str(ann_path)})
            continue

        arr = np.load(prep_path, mmap_mode="r")
        n_signal_epochs = len(arr)

        header_lines, stage_rows = parse_stage_duration_tsv(ann_path)
        annot_duration_s = stage_rows[-1][1]
        n_annot_epochs = int(round(annot_duration_s / args.epoch_sec))

        diff_epochs = n_signal_epochs - n_annot_epochs
        status = "ok"

        if diff_epochs != 0:
            if abs(diff_epochs) <= args.max_fix_epochs:
                status = "would_fix" if not args.apply else "fixed"

                if args.apply:
                    backup_path = ann_path.with_suffix(ann_path.suffix + ".bak_before_length_fix")
                    if not backup_path.exists():
                        shutil.copy2(ann_path, backup_path)

                    target_duration_s = n_signal_epochs * args.epoch_sec
                    new_stage_rows = adjust_stage_rows(stage_rows, target_duration_s)
                    rewrite_stage_duration_tsv(
                        ann_path,
                        header_lines,
                        new_stage_rows,
                        target_duration_s,
                    )
            else:
                status = "large_mismatch_skip"

        rows.append({
            "row": i,
            "status": status,
            "recording_name": row.get("recording_name", ""),
            "mouse_id": row.get("mouse_id", ""),
            "week": row.get("week", ""),
            "segment_id": row.get("segment_id", ""),
            "n_signal_epochs": n_signal_epochs,
            "n_annotation_epochs": n_annot_epochs,
            "diff_epochs_signal_minus_annotation": diff_epochs,
            "annotation_path": str(ann_path),
        })

    report = pd.DataFrame(rows)
    report_path = args.manifest.with_name(args.manifest.stem + "_length_check.csv")
    report.to_csv(report_path, index=False)

    print("\nLength check summary:")
    print(report["status"].value_counts(dropna=False).to_string())

    mismatches = report[report["diff_epochs_signal_minus_annotation"] != 0]
    if len(mismatches):
        print("\nMismatches:")
        cols = [
            "status",
            "mouse_id",
            "week",
            "segment_id",
            "n_signal_epochs",
            "n_annotation_epochs",
            "diff_epochs_signal_minus_annotation",
        ]
        print(mismatches[cols].to_string(index=False))
    else:
        print("\nNo mismatches found.")

    print(f"\nReport written to: {report_path}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to fix small mismatches.")


if __name__ == "__main__":
    main()
