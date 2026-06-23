from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from sleep_loader import (
    parse_exp,
    resolve_associated_path,
    load_bin_segment,
    load_hypnogram_segment,
    default_label_map,
)


def to_naive_utc(dt) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def safe_seconds_diff(later, earlier) -> float:
    return (to_naive_utc(later) - to_naive_utc(earlier)).total_seconds()


def get_status_key_to_label(meta) -> dict[int, str]:
    return {int(s.key): s.label for s in meta.hypnogram_statuses}


def load_one_hypnogram_segment(
    meta,
    seg_idx: int,
    exp_path: Path,
    data_root: Path,
) -> tuple[pd.DataFrame | None, Path]:
    """
    Load exactly one hypnogram segment.
    Returns (None, path) if the hypnogram file is empty.
    """
    hyp_meta = meta.hypnogram_files[seg_idx]
    acq_meta = meta.acquisition_files[seg_idx]

    h_path = resolve_associated_path(exp_path, hyp_meta.filename, data_root=data_root)

    if (not h_path.exists()) or h_path.stat().st_size == 0:
        return None, h_path

    expected_duration_s = hyp_meta.duration_s if hyp_meta.duration_s > 0 else acq_meta.duration_s

    try:
        hyp_seg = load_hypnogram_segment(
            h_path,
            epoch_s=meta.hypnogram_epoch_s,
            tstart_wallclock=hyp_meta.tstart,
            expected_duration_s=expected_duration_s,
            label_map=default_label_map(),
            status_key_to_label=get_status_key_to_label(meta),
        )
    except ValueError as exc:
        if "Empty hypnogram file" in str(exc):
            return None, h_path
        raise

    hyp_seg["segment_id"] = seg_idx
    return hyp_seg, h_path


def infer_channel_map(channel_names: list[str]) -> dict[str, str | None]:
    """
    Infer EEG / EMG / TTL channel names from metadata, independent of mouse ID.
    """
    eeg_name = next((name for name in channel_names if "EEG" in name.upper()), None)
    emg_name = next((name for name in channel_names if "EMG" in name.upper()), None)
    ttl_name = next((name for name in channel_names if "TTL" in name.upper()), None)

    if eeg_name is None or emg_name is None:
        raise ValueError(
            f"Could not infer EEG/EMG channels from {channel_names}. "
            "Expected channel names containing 'EEG' and 'EMG'."
        )

    return {
        "eeg": eeg_name,
        "emg": emg_name,
        "ttl": ttl_name,
    }


def center_u16_to_i16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (x.astype(np.int32) - 32768).astype(np.int16)


def build_native_5s_state_table(
    meta,
    hyp_seg: pd.DataFrame,
    seg_idx: int,
    epoch_s: float = 5.0,
) -> pd.DataFrame:
    """
    Build exact 5 s epochs anchored to the hypnogram start for this segment.
    """
    columns = [
        "epoch_id",
        "segment_id",
        "t0_s",
        "t1_s",
        "label_raw",
        "label_mapped",
        "n_hyp_samples",
        "dominant_fraction",
    ]

    if hyp_seg is None or hyp_seg.empty:
        return pd.DataFrame(columns=columns)

    acq_seg = meta.acquisition_files[seg_idx]
    hyp_file_seg = meta.hypnogram_files[seg_idx]

    offset_s = safe_seconds_diff(hyp_file_seg.tstart, acq_seg.tstart)

    hyp_seg = hyp_seg.sort_values("sample_id_local").reset_index(drop=True)
    n_full_blocks = len(hyp_seg) // 5
    hyp_seg = hyp_seg.iloc[: n_full_blocks * 5].copy()

    rows = []
    for block_id in range(n_full_blocks):
        sub = hyp_seg.iloc[block_id * 5 : (block_id + 1) * 5]

        raw_counts = sub["label_raw"].value_counts(dropna=False)
        mapped_counts = sub["label_mapped"].value_counts(dropna=False)

        t0 = offset_s + block_id * epoch_s
        t1 = t0 + epoch_s

        rows.append(
            {
                "epoch_id": block_id,
                "segment_id": seg_idx,
                "t0_s": float(t0),
                "t1_s": float(t1),
                "label_raw": raw_counts.index[0],
                "label_mapped": mapped_counts.index[0],
                "n_hyp_samples": int(len(sub)),
                "dominant_fraction": float(raw_counts.iloc[0] / raw_counts.sum()),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def add_sample_indices(state_table: pd.DataFrame, fs: float) -> pd.DataFrame:
    out = state_table.copy()
    out["sample_i0"] = np.round(out["t0_s"] * fs).astype(int)
    out["sample_i1"] = np.round(out["t1_s"] * fs).astype(int)
    return out


def build_aligned_training_export(
    eeg_i16: np.ndarray,
    emg_i16: np.ndarray,
    ttl_i16: np.ndarray,
    state_table: pd.DataFrame,
    fs: float,
    epoch_s: float = 5.0,
):
    """
    Build aligned Somnotate training data.

    Preferred mode:
        use the offset-derived sample_i0/sample_i1 from state_table.

    Fallback mode:
        if the derived offset is invalid for this segment, align the training
        export locally to sample 0 and keep as many complete epochs as fit.

    Returns
    -------
    eeg_out, emg_out, ttl_out, aligned_state_table, export_info
    """
    st = state_table.sort_values("epoch_id").reset_index(drop=True).copy()
    if st.empty:
        raise ValueError("state_table is empty")

    samples_per_epoch = int(round(fs * epoch_s))
    raw_n = len(eeg_i16)

    requested_start = int(st.iloc[0]["sample_i0"])
    requested_n_epochs = int(len(st))
    requested_stop = requested_start + requested_n_epochs * samples_per_epoch

    # Preferred: use metadata-based offset if it is valid
    if 0 <= requested_start and requested_stop <= raw_n:
        start_sample = requested_start
        n_epochs = requested_n_epochs
        mode = "metadata_offset"

    else:
        # Fallback: use segment-local alignment
        start_sample = 0
        max_epochs_fit = raw_n // samples_per_epoch
        n_epochs = min(requested_n_epochs, max_epochs_fit)

        if n_epochs <= 0:
            raise ValueError(
                "Cannot build aligned export: no complete epochs fit into signal. "
                f"raw_n={raw_n}, samples_per_epoch={samples_per_epoch}, "
                f"requested_start={requested_start}, requested_n_epochs={requested_n_epochs}"
            )

        st = st.iloc[:n_epochs].copy()
        mode = "fallback_start_at_zero"

    stop_sample = start_sample + n_epochs * samples_per_epoch

    eeg_out = eeg_i16[start_sample:stop_sample]
    emg_out = emg_i16[start_sample:stop_sample]
    ttl_out = ttl_i16[start_sample:stop_sample]

    expected_len = n_epochs * samples_per_epoch
    if len(eeg_out) != expected_len or len(emg_out) != expected_len or len(ttl_out) != expected_len:
        raise ValueError(
            "Aligned signal length mismatch after cropping: "
            f"EEG={len(eeg_out)}, EMG={len(emg_out)}, TTL={len(ttl_out)}, expected={expected_len}"
        )

    # Rebuild the aligned state table so that t=0 corresponds to the first sample
    st = st.reset_index(drop=True).copy()
    st["epoch_id"] = np.arange(len(st), dtype=int)
    st["t0_s"] = st["epoch_id"] * epoch_s
    st["t1_s"] = st["t0_s"] + epoch_s
    st["sample_i0"] = st["epoch_id"] * samples_per_epoch
    st["sample_i1"] = st["sample_i0"] + samples_per_epoch

    export_info = {
        "mode": mode,
        "requested_start_sample": int(requested_start),
        "used_start_sample": int(start_sample),
        "requested_n_epochs": int(requested_n_epochs),
        "used_n_epochs": int(n_epochs),
        "raw_n_samples": int(raw_n),
        "samples_per_epoch": int(samples_per_epoch),
    }

    return eeg_out, emg_out, ttl_out, st, export_info

def find_state_bouts(state_table: pd.DataFrame, state: str = "PS", min_epochs: int = 1) -> pd.DataFrame:
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


def bandpower_from_psd(f: np.ndarray, pxx: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return np.nan
    return float(np.trapz(pxx[mask], f[mask]))


def compute_epoch_features(
    eeg_i16: np.ndarray,
    emg_i16: np.ndarray,
    fs: float,
    state_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in state_table.iterrows():
        i0 = int(row["sample_i0"])
        i1 = int(row["sample_i1"])

        if i0 < 0 or i1 > len(eeg_i16) or i1 <= i0:
            continue

        eeg = eeg_i16[i0:i1].astype(np.float32)
        emg = emg_i16[i0:i1].astype(np.float32)

        if len(eeg) < 32:
            continue

        nperseg = min(len(eeg), int(round(fs * 2)))
        noverlap = min(int(round(fs)), max(0, nperseg // 2))

        f, pxx = signal.welch(
            eeg,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )

        delta = bandpower_from_psd(f, pxx, 1, 4)
        theta = bandpower_from_psd(f, pxx, 6, 9)
        beta = bandpower_from_psd(f, pxx, 15, 30)
        total = bandpower_from_psd(f, pxx, 1, 30)

        emg_rms = float(np.sqrt(np.mean(emg ** 2)))
        emg_abs_mean = float(np.mean(np.abs(emg)))
        theta_delta_ratio = float(theta / (delta + np.finfo(float).eps))

        rows.append(
            {
                "epoch_id": int(row["epoch_id"]),
                "segment_id": int(row["segment_id"]),
                "t0_s": float(row["t0_s"]),
                "t1_s": float(row["t1_s"]),
                "label_raw": row["label_raw"],
                "label_mapped": row["label_mapped"],
                "dominant_fraction": float(row["dominant_fraction"]),
                "delta_1_4": delta,
                "theta_6_9": theta,
                "beta_15_30": beta,
                "total_1_30": total,
                "theta_delta_ratio": theta_delta_ratio,
                "emg_rms": emg_rms,
                "emg_abs_mean": emg_abs_mean,
            }
        )

    return pd.DataFrame(rows)


def map_label_for_somnotate(label_raw: str) -> str:
    mapping = {
        "WK": "Awake",
        "SWS": "NREM",
        "PS": "REM",
        "TR": "Undefined",
        "ND": "Undefined",
        "Artef": "Undefined",
        "Artf": "Undefined",
    }
    return mapping.get(str(label_raw), "Undefined")


def write_visbrain_stage_duration_full(
    state_table: pd.DataFrame,
    recording_duration_s: float,
    out_path: Path,
) -> None:
    """
    Export stage-duration annotation in the original full-recording time frame,
    preserving any offset between signal start and first scored epoch.
    """
    if state_table.empty:
        raise ValueError("state_table is empty")

    st = state_table.sort_values("t0_s").reset_index(drop=True).copy()
    st["som_state"] = st["label_raw"].map(map_label_for_somnotate)

    states = []
    end_times = []

    first_t0 = float(st.iloc[0]["t0_s"])
    if first_t0 > 1e-6:
        states.append("Undefined")
        end_times.append(first_t0)

    current_state = None
    current_end = None

    for _, row in st.iterrows():
        state = row["som_state"]
        t1 = float(row["t1_s"])

        if current_state is None:
            current_state = state
            current_end = t1
        elif state == current_state:
            current_end = t1
        else:
            states.append(current_state)
            end_times.append(current_end)
            current_state = state
            current_end = t1

    if current_state is not None:
        states.append(current_state)
        end_times.append(current_end)

    if abs(end_times[-1] - recording_duration_s) > 1e-6:
        states.append("Undefined")
        end_times.append(recording_duration_s)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"*Duration_sec\t{recording_duration_s:.6f}\n")
        f.write("*Datafile\tUnspecified\n")
        for s, t in zip(states, end_times):
            f.write(f"{s}\t{t:.6f}\n")


def write_visbrain_stage_duration_aligned(
    state_table: pd.DataFrame,
    out_path: Path,
    epoch_s: float = 5.0,
) -> None:
    """
    Export stage-duration annotation in an aligned time frame that starts at t=0
    and has exactly one 5 s bin per row of state_table.
    """
    st = state_table.sort_values("epoch_id").reset_index(drop=True).copy()
    if st.empty:
        raise ValueError("state_table is empty")

    epoch_states = st["label_raw"].map(map_label_for_somnotate).fillna("Undefined").tolist()
    recording_duration_s = len(epoch_states) * epoch_s

    states = []
    end_times = []

    current_state = None
    run_start_idx = 0

    for i, state in enumerate(epoch_states):
        if current_state is None:
            current_state = state
            run_start_idx = i
        elif state != current_state:
            states.append(current_state)
            end_times.append(i * epoch_s)
            current_state = state
            run_start_idx = i

    states.append(current_state)
    end_times.append(recording_duration_s)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"*Duration_sec\t{recording_duration_s:.6f}\n")
        f.write("*Datafile\tUnspecified\n")
        for s, t in zip(states, end_times):
            f.write(f"{s}\t{t:.6f}\n")


def write_epoch_csv_for_somnotate_aligned(state_table: pd.DataFrame, out_path: Path) -> None:
    st = state_table.sort_values("epoch_id").reset_index(drop=True).copy()
    epoch_states = st["label_raw"].map(map_label_for_somnotate).fillna("Undefined")
    epoch_states.to_csv(out_path, index=False, header=False)


def write_edf(
    eeg_i16: np.ndarray,
    emg_i16: np.ndarray,
    ttl_i16: np.ndarray,
    fs: float,
    out_path: Path,
) -> None:
    """
    Export EDF with centered signals in arbitrary units.
    Requires pyedflib.
    """
    import pyedflib

    signals = [
        eeg_i16.astype(np.float64),
        emg_i16.astype(np.float64),
        ttl_i16.astype(np.float64),
    ]
    labels = ["EEG", "EMG", "TTL"]

    channel_info = []
    for label, sig in zip(labels, signals):
        sig_min = float(np.min(sig))
        sig_max = float(np.max(sig))

        if sig_min == sig_max:
            sig_min -= 1.0
            sig_max += 1.0

        channel_info.append(
            {
                "label": label,
                "dimension": "a.u.",
                "sample_frequency": float(fs),
                "physical_min": sig_min,
                "physical_max": sig_max,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        )

    writer = pyedflib.EdfWriter(
        str(out_path),
        len(signals),
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )
    try:
        writer.setSignalHeaders(channel_info)
        writer.writeSamples(signals)
    finally:
        writer.close()


def build_somnotate_manifest_row(seg_dir: Path, fs: float) -> dict:
    """
    Manifest row for aligned supervised Somnotate training/evaluation.
    """
    return {
        "file_path_raw_signals": str(seg_dir / "somnotate_input.edf"),
        "file_path_preprocessed_signals": str(seg_dir / "somnotate_preprocessed.npy"),
        "file_path_manual_state_annotation": str(seg_dir / "somnotate_annotation.tsv"),
        "file_path_automated_state_annotation": str(seg_dir / "somnotate_automated.tsv"),
        "file_path_refined_state_annotation": str(seg_dir / "somnotate_refined.tsv"),
        "file_path_review_intervals": str(seg_dir / "somnotate_review_intervals.csv"),
        "file_path_state_probabilities": str(seg_dir / "somnotate_state_probabilities.npz"),
        "sampling_frequency_in_hz": float(fs),
        "frontal_eeg_signal_label": "EEG",
        "emg_signal_label": "EMG",
    }


def prepare_recording(exp_path: Path, data_root: Path, out_root: Path) -> None:
    meta = parse_exp(exp_path)
    channel_map = infer_channel_map(meta.channel_names)
    rec_out = out_root / exp_path.stem
    rec_out.mkdir(parents=True, exist_ok=True)

    (rec_out / "recording_meta.json").write_text(
        json.dumps(
            {
                "exp_path": str(exp_path),
                "sampling_rate": meta.sampling_rate,
                "channel_names": meta.channel_names,
                "channel_map": channel_map,
                "n_acquisition_segments": len(meta.acquisition_files),
                "n_hyp_segments": len(meta.hypnogram_files),
            },
            indent=2,
        )
    )

    manifest_rows = []
    somnotate_rows = []

    n_segments = min(len(meta.acquisition_files), len(meta.hypnogram_files))

    for seg_idx in range(n_segments):
        seg_dir = rec_out / f"segment_{seg_idx:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        acq_seg = meta.acquisition_files[seg_idx]
        bin_path = resolve_associated_path(exp_path, acq_seg.filename, data_root=data_root)

        raw = load_bin_segment(
            bin_path,
            channel_names=meta.channel_names,
            dtype="<u2",
            header_offset=meta.header_offset,
            interleave="sample",
        )

        eeg_i16 = center_u16_to_i16(raw[channel_map["eeg"]])
        emg_i16 = center_u16_to_i16(raw[channel_map["emg"]])

        if channel_map["ttl"] is not None:
            ttl_i16 = center_u16_to_i16(raw[channel_map["ttl"]])
        else:
            ttl_i16 = np.zeros_like(eeg_i16, dtype=np.int16)

        np.savez_compressed(
            seg_dir / "signals_raw_centered_i16.npz",
            eeg=eeg_i16,
            emg=emg_i16,
            ttl=ttl_i16,
            fs=np.array([meta.sampling_rate], dtype=np.float32),
        )

        hyp_seg, hyp_path = load_one_hypnogram_segment(
            meta=meta,
            seg_idx=seg_idx,
            exp_path=exp_path,
            data_root=data_root,
        )

        if hyp_seg is None or hyp_seg.empty:
            print(f"Skipping segment {seg_idx}: empty hypnogram file ({hyp_path.name})")
            manifest_rows.append(
                {
                    "segment_id": seg_idx,
                    "bin_file": acq_seg.filename,
                    "hyp_file": meta.hypnogram_files[seg_idx].filename,
                    "signal_start": acq_seg.tstart.isoformat(),
                    "hyp_start": meta.hypnogram_files[seg_idx].tstart.isoformat(),
                    "offset_s": safe_seconds_diff(meta.hypnogram_files[seg_idx].tstart, acq_seg.tstart),
                    "n_samples_signal": len(eeg_i16),
                    "n_epochs_5s": 0,
                    "n_ps_bouts": 0,
                    "status": "skipped_empty_hypnogram",
                }
            )
            continue

        state_table = build_native_5s_state_table(meta, hyp_seg, seg_idx, epoch_s=5.0)
        state_table = add_sample_indices(state_table, meta.sampling_rate)

        if state_table.empty:
            print(f"Skipping segment {seg_idx}: state_table is empty")
            manifest_rows.append(
                {
                    "segment_id": seg_idx,
                    "bin_file": acq_seg.filename,
                    "hyp_file": meta.hypnogram_files[seg_idx].filename,
                    "signal_start": acq_seg.tstart.isoformat(),
                    "hyp_start": meta.hypnogram_files[seg_idx].tstart.isoformat(),
                    "offset_s": safe_seconds_diff(meta.hypnogram_files[seg_idx].tstart, acq_seg.tstart),
                    "n_samples_signal": len(eeg_i16),
                    "n_epochs_5s": 0,
                    "n_ps_bouts": 0,
                    "status": "skipped_empty_state_table",
                }
            )
            continue

        ps_bouts = find_state_bouts(state_table, state="PS", min_epochs=2)
        features = compute_epoch_features(eeg_i16, emg_i16, meta.sampling_rate, state_table)

        state_table.to_csv(seg_dir / "state_table_5s.csv", index=False)
        ps_bouts.to_csv(seg_dir / "ps_bouts.csv", index=False)
        features.to_csv(seg_dir / "epoch_features_5s.csv", index=False)

        # Full-length reference files
        write_edf(
            eeg_i16=eeg_i16,
            emg_i16=emg_i16,
            ttl_i16=ttl_i16,
            fs=meta.sampling_rate,
            out_path=seg_dir / "somnotate_input_full.edf",
        )

        write_visbrain_stage_duration_full(
            state_table=state_table,
            recording_duration_s=len(eeg_i16) / meta.sampling_rate,
            out_path=seg_dir / "somnotate_annotation_full.tsv",
        )

        # Aligned training/evaluation files
        eeg_aligned, emg_aligned, ttl_aligned, aligned_state_table, export_info = build_aligned_training_export(
            eeg_i16=eeg_i16,
            emg_i16=emg_i16,
            ttl_i16=ttl_i16,
            state_table=state_table,
            fs=meta.sampling_rate,
            epoch_s=5.0,
        )

        write_edf(
            eeg_i16=eeg_aligned,
            emg_i16=emg_aligned,
            ttl_i16=ttl_aligned,
            fs=meta.sampling_rate,
            out_path=seg_dir / "somnotate_input.edf",
        )

        write_visbrain_stage_duration_aligned(
            state_table=aligned_state_table,
            out_path=seg_dir / "somnotate_annotation.tsv",
            epoch_s=5.0,
        )

        write_epoch_csv_for_somnotate_aligned(
            state_table=aligned_state_table,
            out_path=seg_dir / "somnotate_annotation_epoch.csv",
        )

        (seg_dir / "somnotate_alignment_info.json").write_text(
            json.dumps(export_info, indent=2)
        )

        if export_info["mode"] != "metadata_offset":
            print(
                f"Segment {seg_idx}: used fallback aligned export "
                f"(requested_start_sample={export_info['requested_start_sample']}, "
                f"used_start_sample={export_info['used_start_sample']}, "
                f"used_n_epochs={export_info['used_n_epochs']})"
            )

        somnotate_rows.append(build_somnotate_manifest_row(seg_dir, fs=meta.sampling_rate))

        manifest_rows.append(
            {
                "segment_id": seg_idx,
                "bin_file": acq_seg.filename,
                "hyp_file": meta.hypnogram_files[seg_idx].filename,
                "signal_start": acq_seg.tstart.isoformat(),
                "hyp_start": meta.hypnogram_files[seg_idx].tstart.isoformat(),
                "offset_s": safe_seconds_diff(meta.hypnogram_files[seg_idx].tstart, acq_seg.tstart),
                "n_samples_signal": len(eeg_i16),
                "n_epochs_5s": len(state_table),
                "aligned_n_epochs": export_info["used_n_epochs"],
                "alignment_mode": export_info["mode"],
                "n_ps_bouts": len(ps_bouts),
                "status": "ok",
            }
        )

    pd.DataFrame(manifest_rows).to_csv(rec_out / "segment_manifest.csv", index=False)
    pd.DataFrame(somnotate_rows).to_csv(rec_out / "somnotate_manifest.csv", index=False)

    print(f"Prepared: {rec_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Path to one .exp file")
    parser.add_argument("--data-root", required=True, help="Folder where .bin/.H files live")
    parser.add_argument("--out-root", required=True, help="Output folder for prepared data")
    args = parser.parse_args()

    prepare_recording(
        exp_path=Path(args.exp),
        data_root=Path(args.data_root),
        out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()