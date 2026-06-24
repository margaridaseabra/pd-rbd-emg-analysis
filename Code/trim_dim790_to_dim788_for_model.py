#!/usr/bin/env python3
"""
Create 788-feature compatibility copies from 790-feature Somnotate preprocessed arrays.

This is intended for recordings where the final WT reference model expects 788 features,
but preprocessing produced 790 features.

Assumption:
    790 = 395 EEG-like features + 395 EMG-like features
    788 = 394 EEG-like features + 394 EMG-like features

The script drops the final feature from each half:
    first block:  columns 0:394
    second block: columns 395:789

It does not overwrite the original .npy files.

Example:
    python Code/trim_dim790_to_dim788_for_model.py \
      --input data/manifests/new_weeks_somnotate_FULL_local_dim790.csv \
      --output data/manifests/new_weeks_somnotate_FULL_local_dim790_trimmed_to_dim788.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def suffix_path(path_value, suffix):
    if pd.isna(path_value):
        return path_value
    path_value = str(path_value).strip()
    if path_value == "":
        return path_value
    p = Path(path_value)
    return str(p.with_name(p.stem + suffix + p.suffix))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "file_path_preprocessed_signals" not in df.columns:
        raise ValueError("Input manifest must contain file_path_preprocessed_signals")

    rows = []
    failures = []

    for i, row in df.iterrows():
        try:
            old_path = Path(str(row["file_path_preprocessed_signals"]))
            if not old_path.exists():
                raise FileNotFoundError(old_path)

            arr = np.load(old_path)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")

            n_epochs, n_features = arr.shape

            if n_features != 790:
                raise ValueError(f"Expected 790 features, got {n_features}: {old_path}")

            # Split 790 into 395 + 395, then trim each half to 394.
            first = arr[:, 0:394]
            second = arr[:, 395:789]
            trimmed = np.concatenate([first, second], axis=1)

            if trimmed.shape[1] != 788:
                raise RuntimeError(f"Trimmed array has wrong shape: {trimmed.shape}")

            new_path = old_path.with_name(old_path.stem + "_trimmed_to_788" + old_path.suffix)
            np.save(new_path, trimmed)

            new_row = row.copy()
            new_row["file_path_preprocessed_signals_original_790"] = str(old_path)
            new_row["file_path_preprocessed_signals"] = str(new_path)
            new_row["n_features_original"] = 790
            new_row["n_features_trimmed"] = 788
            new_row["feature_trim_note"] = "Dropped final column from each 395-feature modality block to match 788-feature model."

            # Avoid overwriting any existing 790 outputs.
            for col in [
                "file_path_automated_state_annotation",
                "file_path_review_intervals",
                "file_path_state_probabilities",
            ]:
                if col in new_row:
                    new_row[col] = suffix_path(new_row[col], "_trimmed788")

            rows.append(new_row)

            print(f"{i}: OK {arr.shape} -> {trimmed.shape} | {new_path}")

        except Exception as exc:
            print(f"{i}: ERROR {repr(exc)}")
            failures.append({**row.to_dict(), "error": repr(exc)})

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print("\nWrote trimmed manifest:")
    print(args.output)
    print("Rows:", len(out_df))

    if failures:
        fail_path = args.output.with_name(args.output.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print("\nFailures written to:")
        print(fail_path)


if __name__ == "__main__":
    main()
