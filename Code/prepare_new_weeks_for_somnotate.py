#!/usr/bin/env python3
"""
Prepare week 5/8/10 recordings for Somnotate.

Reads a local manifest with exp_path entries, runs prepare_one_recording.py logic,
and builds one combined Somnotate manifest.

Example:
    python Code/prepare_new_weeks_for_somnotate.py \
      --manifest data/manifests/new_weeks_manifest_local.csv \
      --out-root "/Volumes/T7/Margarida/RBD-KatiasData/processed_new_weeks/somnotate_prepared" \
      --combined-out data/manifests/new_weeks_somnotate_FULL_local.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_one_recording import prepare_recording


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--combined-out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    if "exp_path" not in df.columns:
        raise ValueError("Manifest must contain an exp_path column.")

    args.out_root.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    failed_rows = []

    for i, row in df.iterrows():
        exp_path = Path(str(row["exp_path"])).expanduser()

        if not exp_path.exists():
            print(f"[{i}] SKIP missing exp: {exp_path}")
            failed_rows.append({**row.to_dict(), "error": "missing exp_path"})
            continue

        data_root = exp_path.parent

        print(f"\n[{i}] Preparing:")
        print(f"  exp:       {exp_path}")
        print(f"  data_root: {data_root}")

        try:
            prepare_recording(
                exp_path=exp_path,
                data_root=data_root,
                out_root=args.out_root,
            )

            rec_out = args.out_root / exp_path.stem
            som_manifest = rec_out / "somnotate_manifest.csv"

            if not som_manifest.exists():
                raise FileNotFoundError(f"Missing output manifest: {som_manifest}")

            sdf = pd.read_csv(som_manifest)

            for col in df.columns:
                if col not in sdf.columns:
                    sdf[col] = row[col]

            sdf["prepared_recording_dir"] = str(rec_out)
            sdf["source_exp_path"] = str(exp_path)

            combined_rows.append(sdf)

            print(f"  OK: {len(sdf)} segments")

        except Exception as exc:
            print(f"  ERROR: {repr(exc)}")
            failed_rows.append({**row.to_dict(), "error": repr(exc)})

    if combined_rows:
        out_df = pd.concat(combined_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame()

    args.combined_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.combined_out, index=False)

    print("\nWrote combined Somnotate manifest:")
    print(f"  {args.combined_out}")
    print(f"Rows: {len(out_df)}")

    if len(out_df):
        print("\nSummary by group/week:")
        print(out_df.groupby(["group", "week"], dropna=False).size().reset_index(name="n_segments").to_string(index=False))

    if failed_rows:
        fail_path = args.combined_out.with_name(args.combined_out.stem + "_failures.csv")
        pd.DataFrame(failed_rows).to_csv(fail_path, index=False)
        print("\nFailures written to:")
        print(f"  {fail_path}")


if __name__ == "__main__":
    main()
