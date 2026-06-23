from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from sleep_loader import (
    parse_exp,
    summarise_experiment,
    resolve_associated_path,
    infer_hypnogram_format,
    load_single_recording_segment,
    load_hypnogram_segment,
    default_label_map,
)


# ---------- CONFIG ----------
DATA_ROOT = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/converted/"
)
OUT_DIR = DATA_ROOT / "analysis_outputs"
BIN_DTYPE = "<u2"
INTERLEAVE = "sample"
MIN_PS_EPOCHS = 2
# ---------------------------


def discover_exp_files(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*.exp"))


def add_hyp_global_times(hyp: pd.DataFrame, signal_ref: datetime) -> pd.DataFrame:
    hyp = hyp.copy()
    hyp["t0_s_global"] = (
        (hyp["tstart_wallclock"] - signal_ref).dt.total_seconds() + hyp["t0_s_local"]
    )
    hyp["t1_s_global"] = (
        (hyp["tstart_wallclock"] - signal_ref).dt.total_seconds() + hyp["t1_s_local"]
    )
    return hyp


def constant_5_block_fraction(hyp_seg: pd.DataFrame) -> float:
    if len(hyp_seg) < 5:
        return np.nan

    nunique_vals = []
    for i in range(0, len(hyp_seg) - 4, 5):
        block = hyp_seg["label_raw"].iloc[i:i + 5]
        nunique_vals.append(block.nunique())

    if len(nunique_vals) == 0:
        return np.nan

    nunique_vals = np.asarray(nunique_vals)
    return float(np.mean(nunique_vals == 1))


def collapse_hyp_to_native_5s(hyp_df: pd.DataFrame, epoch_s: float = 5.0) -> pd.DataFrame:
    """
    Build exact 5.0 s epochs anchored to the first hypnogram sample of each segment.
    """
    rows = []
    epoch_id_global = 0

    for seg_id, seg in hyp_df.groupby("segment_id", sort=True):
        seg = seg.sort_values("sample_id_local").reset_index(drop=True)

        n_full_blocks = len(seg) // 5
        if n_full_blocks == 0:
            continue

        seg = seg.iloc[: n_full_blocks * 5].copy()
        seg_start = float(seg["t0_s_global"].iloc[0])

        for block_id in range(n_full_blocks):
            sub = seg.iloc[block_id * 5 : (block_id + 1) * 5]

            raw_counts = sub["label_raw"].value_counts(dropna=False)
            mapped_counts = sub["label_mapped"].value_counts(dropna=False)

            t0 = seg_start + block_id * epoch_s
            t1 = t0 + epoch_s

            rows.append(
                {
                    "epoch_id": epoch_id_global,
                    "segment_id": int(seg_id),
                    "block_id_local": int(block_id),
                    "t0_s": float(t0),
                    "t1_s": float(t1),
                    "label_raw": raw_counts.index[0],
                    "label_mapped": mapped_counts.index[0],
                    "n_hyp_samples": int(len(sub)),
                    "dominant_fraction": float(raw_counts.iloc[0] / raw_counts.sum()),
                }
            )
            epoch_id_global += 1

    return pd.DataFrame(rows)


def add_local_sample_indices(
    state_table: pd.DataFrame,
    segment_signal_start_global_s: float,
    fs: float,
) -> pd.DataFrame:
    out = state_table.copy()
    out["sample_i0_local"] = np.round((out["t0_s"] - segment_signal_start_global_s) * fs).astype(int)
    out["sample_i1_local"] = np.round((out["t1_s"] - segment_signal_start_global_s) * fs).astype(int)
    return out


def find_state_bouts(
    state_table: pd.DataFrame,
    state: str = "PS",
    min_epochs: int = 1,
) -> pd.DataFrame:
    is_state = state_table["label_raw"].eq(state).fillna(False).to_numpy()

    bouts = []
    start_idx = None

    for i, val in enumerate(is_state):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            end_idx = i - 1
            if (end_idx - start_idx + 1) >= min_epochs:
                bouts.append((start_idx, end_idx))
            start_idx = None

    if start_idx is not None:
        end_idx = len(is_state) - 1
        if (end_idx - start_idx + 1) >= min_epochs:
            bouts.append((start_idx, end_idx))

    rows = []
    for bout_id, (i0, i1) in enumerate(bouts):
        row0 = state_table.iloc[i0]
        row1 = state_table.iloc[i1]

        prev_label = state_table.iloc[i0 - 1]["label_raw"] if i0 > 0 else None
        next_label = state_table.iloc[i1 + 1]["label_raw"] if i1 < len(state_table) - 1 else None

        rows.append(
            {
                "bout_id": bout_id,
                "state": state,
                "start_epoch": int(i0),
                "end_epoch": int(i1),
                "start_s": float(row0["t0_s"]),
                "end_s": float(row1["t1_s"]),
                "duration_s": float(row1["t1_s"] - row0["t0_s"]),
                "n_epochs": int(i1 - i0 + 1),
                "prev_label": prev_label,
                "next_label": next_label,
            }
        )

    return pd.DataFrame(rows)


def build_segment_pair_table(meta) -> pd.DataFrame:
    n = min(len(meta.acquisition_files), len(meta.hypnogram_files))
    rows = []
    for seg_idx in range(n):
        acq = meta.acquisition_files[seg_idx]
        hyp = meta.hypnogram_files[seg_idx]
        rows.append(
            {
                "segment_id": seg_idx,
                "acq_file": acq.filename,
                "hyp_file": hyp.filename,
                "acq_start": acq.tstart.isoformat(),
                "hyp_start": hyp.tstart.isoformat(),
                "offset_s": (hyp.tstart - acq.tstart).total_seconds(),
                "acq_duration_s": acq.duration_s,
                "hyp_duration_s": hyp.duration_s,
            }
        )
    return pd.DataFrame(rows)


def analyze_segment(
    meta,
    seg_idx: int,
    data_root: Path,
    signal_ref: datetime,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    loaded_seg = load_single_recording_segment(
        meta,
        seg_idx=seg_idx,
        data_root=data_root,
        dtype=BIN_DTYPE,
        interleave=INTERLEAVE,
    )

    acq_seg = meta.acquisition_files[seg_idx]
    hyp_seg = meta.hypnogram_files[seg_idx]

    segment_signal_start_global_s = (acq_seg.tstart - signal_ref).total_seconds()

    hyp_path = resolve_associated_path(meta.exp_path, hyp_seg.filename, data_root=data_root)
    hyp_info = infer_hypnogram_format(
        hyp_path,
        known_label_keys={int(s.key) for s in meta.hypnogram_statuses},
        expected_duration_s=hyp_seg.duration_s,
        epoch_s=meta.hypnogram_epoch_s,
    )

    status_key_to_label = {int(s.key): s.label for s in meta.hypnogram_statuses}
    hyp_df = load_hypnogram_segment(
        hyp_path,
        epoch_s=meta.hypnogram_epoch_s or 5.0,
        tstart_wallclock=hyp_seg.tstart,
        expected_duration_s=hyp_seg.duration_s,
        label_map=label_map,
        status_key_to_label=status_key_to_label,
    )
    hyp_df["segment_id"] = seg_idx
    hyp_df = add_hyp_global_times(hyp_df, signal_ref=signal_ref)

    state_table = collapse_hyp_to_native_5s(hyp_df, epoch_s=meta.hypnogram_epoch_s or 5.0)
    state_table = add_local_sample_indices(
        state_table,
        segment_signal_start_global_s=segment_signal_start_global_s,
        fs=meta.sampling_rate,
    )

    ps_bouts = find_state_bouts(state_table, state="PS", min_epochs=MIN_PS_EPOCHS)

    qc_flags = []
    offset_s = (hyp_seg.tstart - acq_seg.tstart).total_seconds()
    const5 = constant_5_block_fraction(hyp_df)

    if abs(offset_s) > 10:
        qc_flags.append("large_offset")
    if abs(len(next(iter(loaded_seg.data.values()))) / meta.sampling_rate - acq_seg.duration_s) > 0.5:
        qc_flags.append("signal_duration_mismatch")
    if pd.notna(const5) and const5 < 0.90:
        qc_flags.append("weak_5s_repeat_pattern")

    qc_row = {
        "exp_name": meta.exp_path.stem,
        "segment_id": seg_idx,
        "acq_file": acq_seg.filename,
        "hyp_file": hyp_seg.filename,
        "signal_start": acq_seg.tstart.isoformat(),
        "hyp_start": hyp_seg.tstart.isoformat(),
        "offset_s": offset_s,
        "signal_duration_declared_s": acq_seg.duration_s,
        "signal_duration_loaded_s": len(next(iter(loaded_seg.data.values()))) / meta.sampling_rate,
        "hyp_duration_declared_s": hyp_seg.duration_s,
        "hyp_n_samples": int(len(hyp_df)),
        "hyp_dt_est_s": float(hyp_df["dt_est_s"].iloc[0]),
        "hyp_dt_used_s": float(hyp_df["dt_used_s"].iloc[0]),
        "constant_5_block_fraction": const5,
        "first_native_epoch_start_s": float(state_table.iloc[0]["t0_s"]) if len(state_table) else np.nan,
        "n_state_epochs": int(len(state_table)),
        "n_ps_bouts": int(len(ps_bouts)),
        "qc_flag": ";".join(qc_flags),
    }

    state_table = state_table.copy()
    state_table["exp_name"] = meta.exp_path.stem

    ps_bouts = ps_bouts.copy()
    ps_bouts["exp_name"] = meta.exp_path.stem
    ps_bouts["segment_id"] = seg_idx

    return state_table, ps_bouts, qc_row


def analyze_experiment(exp_path: Path, data_root: Path) -> dict[str, pd.DataFrame]:
    meta = parse_exp(exp_path)
    signal_ref = meta.acquisition_files[0].tstart
    label_map = default_label_map()

    n = min(len(meta.acquisition_files), len(meta.hypnogram_files))

    state_tables = []
    ps_bouts_all = []
    qc_rows = []

    for seg_idx in range(n):
        state_table, ps_bouts, qc_row = analyze_segment(
            meta=meta,
            seg_idx=seg_idx,
            data_root=data_root,
            signal_ref=signal_ref,
            label_map=label_map,
        )
        state_tables.append(state_table)
        ps_bouts_all.append(ps_bouts)
        qc_rows.append(qc_row)

    return {
        "meta_summary": pd.DataFrame([summarise_experiment(meta)]),
        "segment_pairs": build_segment_pair_table(meta),
        "qc": pd.DataFrame(qc_rows),
        "state_tables": pd.concat(state_tables, ignore_index=True) if state_tables else pd.DataFrame(),
        "ps_bouts": pd.concat(ps_bouts_all, ignore_index=True) if ps_bouts_all else pd.DataFrame(),
    }


def analyze_all_recordings(data_root: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_files = discover_exp_files(data_root)
    all_qc = []
    all_pairs = []
    all_states = []
    all_bouts = []
    errors = []

    for exp_path in exp_files:
        try:
            result = analyze_experiment(exp_path, data_root=data_root)
            all_qc.append(result["qc"])
            all_pairs.append(result["segment_pairs"])
            all_states.append(result["state_tables"])
            all_bouts.append(result["ps_bouts"])
            print(f"OK: {exp_path.name}")
        except Exception as exc:
            errors.append({"exp_name": exp_path.stem, "error": repr(exc)})
            print(f"ERROR: {exp_path.name} -> {exc}")

    qc_df = pd.concat(all_qc, ignore_index=True) if all_qc else pd.DataFrame()
    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    states_df = pd.concat(all_states, ignore_index=True) if all_states else pd.DataFrame()
    bouts_df = pd.concat(all_bouts, ignore_index=True) if all_bouts else pd.DataFrame()
    errors_df = pd.DataFrame(errors)

    qc_df.to_csv(out_dir / "qc_summary.csv", index=False)
    pairs_df.to_csv(out_dir / "segment_pairs.csv", index=False)
    states_df.to_csv(out_dir / "state_tables_5s.csv", index=False)
    bouts_df.to_csv(out_dir / "ps_bouts.csv", index=False)
    errors_df.to_csv(out_dir / "errors.csv", index=False)

    return {
        "qc": qc_df,
        "segment_pairs": pairs_df,
        "state_tables": states_df,
        "ps_bouts": bouts_df,
        "errors": errors_df,
    }


def main():
    result = analyze_all_recordings(DATA_ROOT, OUT_DIR)

    print("\n=== QC SUMMARY HEAD ===")
    print(result["qc"].head())

    print("\n=== ERRORS ===")
    print(result["errors"])


if __name__ == "__main__":
    main()