#!/usr/bin/env python3
"""
Create an EEG-only Somnotate manifest from a FULL EEG+EMG manifest.

This script:
- loads each FULL preprocessed .npy file
- splits the features into two equal halves
- saves only the EEG half as a new _EEGonly.npy file
- updates file_path_preprocessed_signals
- updates probability/annotation output paths so EEG-only outputs do not overwrite FULL outputs

Example:
    python Code/make_new_weeks_EEGonly_manifest.py \
      --input data/manifests/new_weeks_somnotate_FULL_local_dim788.csv \
      --output data/manifests/new_weeks_somnotate_EEGonly_local_dim788.csv \
      --eeg-block first
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def suffix_path(path_value: str, suffix: str) -> str:
    path = Path(str(path_value))
    return str(path.with_name(path.stem + suffix + path.suffix))


def make_eegonly_file(path_value: str, eeg_block: str):
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}: {path}")

    n_epochs, n_features = arr.shape

    if n_features % 2 != 0:
        raise ValueError(
            f"Feature dimension is not divisible by 2: {path}, shape={arr.shape}"
        )

    half = n_features // 2

    if eeg_block == "first":
        eeg = arr[:, :half]
    elif eeg_block == "second":
        eeg = arr[:, half:]
    else:
        raise ValueError("--eeg-block must be first or second")

    out = path.with_name(path.stem + "_EEGonly" + path.suffix)
    np.save(out, eeg)

    return out, eeg.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--eeg-block",
        choices=["first", "second"],
        default="first",
        help="Which half of the FULL feature array corresponds to EEG. Default: first.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "file_path_preprocessed_signals" not in df.columns:
        raise ValueError("Input manifest must contain file_path_preprocessed_signals")

    rows = []
    failures = []

    for i, row in df.iterrows():
        try:
            old_pre = row["file_path_preprocessed_signals"]
            new_pre, shape = make_eegonly_file(old_pre, args.eeg_block)

            row = row.copy()
            row["file_path_preprocessed_signals_FULL_EEG_EMG"] = old_pre
            row["file_path_preprocessed_signals"] = str(new_pre)
            row["n_epochs_EEGonly"] = shape[0]
            row["n_features_EEGonly"] = shape[1]
            row["eegonly_source_feature_block"] = args.eeg_block

            # Prevent EEG-only outputs from overwriting FULL outputs.
            for col in [
                "file_path_automated_state_annotation",
                "file_path_review_intervals",
                "file_path_state_probabilities",
            ]:
                if col in row and isinstance(row[col], str) and row[col].strip():
                    row[col] = suffix_path(row[col], "_EEGonly")

            rows.append(row)

            print(
                f"{i}: OK | mouse={row.get('mouse_id', '')} "
                f"week={row.get('week', '')} segment={row.get('segment_id', '')} "
                f"EEGonly_shape={shape}"
            )

        except Exception as exc:
            print(f"{i}: ERROR | {repr(exc)}")
            failures.append({**row.to_dict(), "error": repr(exc)})

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print("\nWrote EEG-only manifest:")
    print(args.output)
    print("Rows:", len(out_df))

    if len(out_df):
        print("\nEEG-only feature dimensions:")
        print(out_df["n_features_EEGonly"].value_counts().to_string())

    if failures:
        fail_path = args.output.with_name(args.output.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print("\nFailures written to:")
        print(fail_path)


if __name__ == "__main__":
    main()
