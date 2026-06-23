#!/usr/bin/env python3
"""
Python spindle-detection pipeline for Katia PD/RBD project.

Purpose
-------
Batch-detect NREM sleep spindles from EEG, summarize spindle metrics per
mouse/week, and optionally merge those metrics with the existing EMG/RBD mouse
metrics exported by the Streamlit QC app.

Typical use from Project_PD_RBD_Katia folder
--------------------------------------------
python spindle_pipeline.py \
  --segments-csv Data/prepared_data/manifests/EMG_burst_detection_NREM_baseline/qc_ready/EMG_episodes_NREMbaseline_qc_ready.csv \
  --out-dir Data/prepared_data/manifests/spindle_detection \
  --merge-rbd-metrics EMG_burst_QC_outputs/live_mouse_level_EMG_REM_metrics.csv

If you do not have a mouse metrics CSV yet, omit --merge-rbd-metrics.

Inputs expected
---------------
The segments CSV can be the EMG event table or a proper segment manifest, as
long as it contains columns like:
    recording_name, group, week, mouse_id, segment_id,
    file_path_raw_signals, file_path_manual_state_annotation

file_path_raw_signals should point to an EDF containing EEG and EMG channels.
file_path_manual_state_annotation should point to a 5 s hypnogram/state file.

Outputs
-------
spindle_events.csv
spindle_segment_metrics.csv
spindle_mouse_week_metrics.csv
mouse_week_metrics_with_spindles.csv, if --merge-rbd-metrics is supplied

Detector summary
----------------
This is a Python analogue of your MATLAB detector, but implemented with standard
SciPy operations:
    - load EEG
    - resample to target Fs, default 200 Hz
    - restrict thresholding and validation to NREM mask
    - sigma band envelope from 10-14 Hz using Hilbert transform
    - broad spindle-band waveform from 9-16 Hz for cycle and amplitude features
    - hysteresis detection: high threshold for event start, low threshold for event boundaries
    - duration/cycle/artifact/spectral-dominance filters

The exact numbers may not be bit-identical to the MATLAB wavelet detector, but
the biological output tables and metrics are directly usable for your app/story.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    from pyedflib import EdfReader
except Exception:  # pragma: no cover
    EdfReader = None


EPOCH_SEC_DEFAULT = 5.0
STATE_NORMALIZATION = {
    "Wake": "Awake",
    "WK": "Awake",
    "W": "Awake",
    "wake": "Awake",
    "WAKE": "Awake",
    "AWAKE": "Awake",
    "Awake": "Awake",
    "NREM": "NREM",
    "Nrem": "NREM",
    "SWS": "NREM",
    "REM": "REM",
    "Rem": "REM",
    "PS": "REM",
    "TR": "Undefined",
    "ND": "Undefined",
    "Undefined": "Undefined",
    "nan": "Undefined",
    "NaN": "Undefined",
    "": "Undefined",
}


@dataclass
class SpindleConfig:
    target_fs: float = 200.0
    epoch_sec: float = EPOCH_SEC_DEFAULT
    sigma_band_low: float = 10.0
    sigma_band_high: float = 14.0
    broad_band_low: float = 9.0
    broad_band_high: float = 16.0
    margin_low: tuple[float, float] = (6.0, 8.5)
    margin_high: tuple[float, float] = (16.5, 20.0)
    envelope_smooth_sec: float = 0.20
    thr_low_z: float = 1.0
    thr_high_z: float = 3.0
    thr_max_z: float = 20.0
    min_duration_sec: float = 0.40
    max_duration_sec: float = 2.00
    min_cycles: int = 5
    max_cycles: int = 30
    min_mask_fraction: float = 0.80
    min_nrem_min_for_threshold: float = 2.0
    artifact_eeg_robust_z: float = 10.0
    artifact_emg_robust_z: float = 8.0
    merge_gap_sec: float = 0.10
    pre_transition_windows_sec: tuple[int, ...] = (30, 60)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def normalize_state(x: object) -> str:
    x = str(x).strip()
    return STATE_NORMALIZATION.get(x, x)


def robust_z(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return (x - med) / scale


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def first_existing_col(row_or_df, candidates: Iterable[str]) -> Optional[str]:
    cols = row_or_df.index if hasattr(row_or_df, "index") else row_or_df.columns
    for c in candidates:
        if c in cols:
            return c
    return None


def infer_channel_indices(labels: list[str]) -> tuple[int, Optional[int], str, Optional[str]]:
    eeg_idx = None
    emg_idx = None
    for i, lab in enumerate(labels):
        u = str(lab).upper()
        if eeg_idx is None and "EEG" in u:
            eeg_idx = i
        if emg_idx is None and "EMG" in u:
            emg_idx = i
    if eeg_idx is None:
        # Fall back to first channel if EDF labels are not helpful.
        eeg_idx = 0
    eeg_label = labels[eeg_idx]
    emg_label = labels[emg_idx] if emg_idx is not None else None
    return eeg_idx, emg_idx, eeg_label, emg_label


def load_edf_eeg_emg(path: str | Path) -> tuple[np.ndarray, Optional[np.ndarray], float, str, Optional[str]]:
    if EdfReader is None:
        raise ImportError("pyedflib is not installed. Install with: pip install pyedflib")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with EdfReader(str(path)) as reader:
        labels = reader.getSignalLabels()
        eeg_idx, emg_idx, eeg_label, emg_label = infer_channel_indices(labels)
        fs = float(reader.getSampleFrequency(eeg_idx))
        eeg = np.asarray(reader.readSignal(eeg_idx), dtype=float)
        emg = None
        if emg_idx is not None:
            emg = np.asarray(reader.readSignal(emg_idx), dtype=float)
            emg_fs = float(reader.getSampleFrequency(emg_idx))
            if abs(emg_fs - fs) > 1e-6:
                # Resample EMG to EEG sampling rate.
                n = int(round(len(eeg)))
                emg = signal.resample(emg, n)
        return eeg, emg, fs, eeg_label, emg_label


# -----------------------------------------------------------------------------
# State annotation loading and bout/transition context
# -----------------------------------------------------------------------------

def load_stage_duration(path: str | Path, n_epochs: Optional[int] = None, epoch_sec: float = EPOCH_SEC_DEFAULT) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    states: list[str] = []
    if not lines:
        return np.array([], dtype=object)

    if lines[0].startswith("*Duration"):
        prev_end_sec = 0.0
        # Many exported files have header lines; app logic starts at lines[2:].
        for line in lines[2:]:
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue
            label = " ".join(parts[:-1])
            end_sec = safe_float(parts[-1])
            if not np.isfinite(end_sec):
                continue
            start_epoch = int(round(prev_end_sec / epoch_sec))
            end_epoch = int(round(end_sec / epoch_sec))
            states.extend([normalize_state(label)] * max(0, end_epoch - start_epoch))
            prev_end_sec = end_sec
    else:
        states = [normalize_state(x) for x in lines]

    if n_epochs is not None:
        if len(states) < n_epochs:
            states.extend(["Undefined"] * (n_epochs - len(states)))
        elif len(states) > n_epochs:
            states = states[:n_epochs]

    return np.asarray(states, dtype=object)


def states_to_sample_mask(states: np.ndarray, target_state: str, n_samples: int, fs: float, epoch_sec: float) -> np.ndarray:
    epochs_per_state = int(round(epoch_sec * fs))
    mask_epoch = np.array([normalize_state(s) == target_state for s in states], dtype=bool)
    mask = np.repeat(mask_epoch, epochs_per_state)
    if len(mask) < n_samples:
        mask = np.pad(mask, (0, n_samples - len(mask)), constant_values=False)
    return mask[:n_samples]


def epoch_index_at_time(t_sec: float, epoch_sec: float) -> int:
    return int(max(0, math.floor(t_sec / epoch_sec)))


def transition_context_for_time(states: np.ndarray, t_sec: float, epoch_sec: float) -> dict:
    """Return distance/context to previous and next state transitions."""
    n = len(states)
    if n == 0:
        return {
            "state_at_event": "Undefined",
            "prev_transition_time_sec": np.nan,
            "next_transition_time_sec": np.nan,
            "distance_to_previous_transition_sec": np.nan,
            "distance_to_next_transition_sec": np.nan,
            "distance_to_nearest_transition_sec": np.nan,
            "previous_state": "",
            "next_state": "",
            "next_transition_type": "",
        }
    idx = min(n - 1, epoch_index_at_time(t_sec, epoch_sec))
    states_norm = np.array([normalize_state(s) for s in states], dtype=object)
    cur = states_norm[idx]
    changes = np.where(states_norm[1:] != states_norm[:-1])[0] + 1
    prev_changes = changes[changes <= idx]
    next_changes = changes[changes > idx]

    if len(prev_changes):
        prev_idx = int(prev_changes[-1])
        prev_time = prev_idx * epoch_sec
        prev_state = states_norm[prev_idx - 1] if prev_idx > 0 else ""
    else:
        prev_idx = None
        prev_time = np.nan
        prev_state = ""

    if len(next_changes):
        next_idx = int(next_changes[0])
        next_time = next_idx * epoch_sec
        next_state = states_norm[next_idx]
        next_transition_type = f"{cur}_to_{next_state}"
    else:
        next_idx = None
        next_time = np.nan
        next_state = ""
        next_transition_type = ""

    d_prev = t_sec - prev_time if np.isfinite(prev_time) else np.nan
    d_next = next_time - t_sec if np.isfinite(next_time) else np.nan
    candidates = [d for d in [d_prev, d_next] if np.isfinite(d)]
    d_near = min(candidates) if candidates else np.nan

    return {
        "state_at_event": cur,
        "prev_transition_time_sec": prev_time,
        "next_transition_time_sec": next_time,
        "distance_to_previous_transition_sec": d_prev,
        "distance_to_next_transition_sec": d_next,
        "distance_to_nearest_transition_sec": d_near,
        "previous_state": prev_state,
        "next_state": next_state,
        "next_transition_type": next_transition_type,
    }


def nrem_bouts_from_states(states: np.ndarray, epoch_sec: float) -> list[dict]:
    states_norm = np.array([normalize_state(s) for s in states], dtype=object)
    nrem = states_norm == "NREM"
    if len(nrem) == 0:
        return []
    starts = np.where(nrem & np.r_[True, ~nrem[:-1]])[0]
    ends = np.where(nrem & np.r_[~nrem[1:], True])[0]
    bouts = []
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        next_state = states_norm[e + 1] if e + 1 < len(states_norm) else ""
        prev_state = states_norm[s - 1] if s > 0 else ""
        bouts.append({
            "nrem_bout_id": i,
            "nrem_start_sec": float(s * epoch_sec),
            "nrem_end_sec": float((e + 1) * epoch_sec),
            "nrem_duration_sec": float((e - s + 1) * epoch_sec),
            "previous_state_before_NREM": prev_state,
            "next_state_after_NREM": next_state,
        })
    return bouts


def add_nrem_bout_context(row: dict, bouts: list[dict]) -> dict:
    mid = 0.5 * (row["start_sec"] + row["end_sec"])
    for b in bouts:
        if b["nrem_start_sec"] <= mid < b["nrem_end_sec"]:
            out = dict(b)
            out["time_from_nrem_start_sec"] = mid - b["nrem_start_sec"]
            out["time_to_nrem_end_sec"] = b["nrem_end_sec"] - mid
            return out
    return {
        "nrem_bout_id": np.nan,
        "nrem_start_sec": np.nan,
        "nrem_end_sec": np.nan,
        "nrem_duration_sec": np.nan,
        "previous_state_before_NREM": "",
        "next_state_after_NREM": "",
        "time_from_nrem_start_sec": np.nan,
        "time_to_nrem_end_sec": np.nan,
    }


# -----------------------------------------------------------------------------
# Signal processing and detection
# -----------------------------------------------------------------------------

def butter_bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if high >= nyq:
        high = nyq - 0.5
    sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def smooth_moving_average(x: np.ndarray, fs: float, win_sec: float) -> np.ndarray:
    win = max(1, int(round(win_sec * fs)))
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def find_hysteresis_events(power_z: np.ndarray, low_thr: float, high_thr: float, fs: float, merge_gap_sec: float) -> list[tuple[int, int]]:
    above_high = power_z >= high_thr
    if not np.any(above_high):
        return []
    high_starts = np.where(above_high & np.r_[True, ~above_high[:-1]])[0]
    high_ends = np.where(above_high & np.r_[~above_high[1:], True])[0]
    events = []
    for hs, he in zip(high_starts, high_ends):
        # expand to low threshold boundary
        s = hs
        while s > 0 and power_z[s] >= low_thr:
            s -= 1
        e = he
        while e < len(power_z) - 1 and power_z[e] >= low_thr:
            e += 1
        events.append((s, e))

    # merge close events
    if not events:
        return []
    events = sorted(events)
    merged = [events[0]]
    max_gap = int(round(merge_gap_sec * fs))
    for s, e in events[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def psd_peak_frequency(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[float, float, float]:
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    nperseg = min(len(x), max(16, int(round(2 * fs))))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=0)
    weighted_pxx = f * pxx
    keep = (f >= band[0]) & (f <= band[1])
    if not np.any(keep):
        return np.nan, np.nan, np.nan
    peak_idx = np.argmax(weighted_pxx[keep])
    f_keep = f[keep]
    p_keep = weighted_pxx[keep]
    return float(f_keep[peak_idx]), float(np.max(p_keep)), float(np.nanmean(p_keep))


def detect_spindles_one_signal(
    eeg: np.ndarray,
    fs: float,
    states: np.ndarray,
    cfg: SpindleConfig,
    emg: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict]:
    """Detect spindles in one EEG segment and return event rows plus diagnostic params."""
    eeg = np.asarray(eeg, dtype=float)
    eeg = signal.detrend(eeg)

    target_fs = float(cfg.target_fs)
    if abs(fs - target_fs) > 1e-6:
        n_target = int(round(len(eeg) * target_fs / fs))
        eeg_rs = signal.resample(eeg, n_target)
        emg_rs = signal.resample(np.asarray(emg, dtype=float), n_target) if emg is not None else None
    else:
        eeg_rs = eeg
        emg_rs = np.asarray(emg, dtype=float) if emg is not None else None

    n = len(eeg_rs)
    n_epochs = int(math.ceil((n / target_fs) / cfg.epoch_sec))
    if len(states) < n_epochs:
        states = np.pad(states, (0, n_epochs - len(states)), constant_values="Undefined")
    elif len(states) > n_epochs:
        states = states[:n_epochs]

    nrem_mask = states_to_sample_mask(states, "NREM", n, target_fs, cfg.epoch_sec)
    nrem_min = nrem_mask.sum() / target_fs / 60
    if nrem_min < cfg.min_nrem_min_for_threshold:
        return pd.DataFrame(), {"reason": "too_little_NREM", "nrem_min": nrem_min}

    # Sigma envelope for detection.
    sigma = butter_bandpass_filter(eeg_rs, target_fs, cfg.sigma_band_low, cfg.sigma_band_high)
    envelope = np.abs(signal.hilbert(sigma))
    envelope = smooth_moving_average(envelope, target_fs, cfg.envelope_smooth_sec)

    # Robust thresholding estimated only from NREM samples.
    env_nrem = envelope[nrem_mask]
    env_z_full = robust_z(envelope)
    env_z_nrem = robust_z(env_nrem)
    nrem_med = np.nanmedian(env_nrem)
    nrem_mad = np.nanmedian(np.abs(env_nrem - nrem_med))
    nrem_scale = 1.4826 * nrem_mad
    if not np.isfinite(nrem_scale) or nrem_scale == 0:
        nrem_scale = np.nanstd(env_nrem)
    if not np.isfinite(nrem_scale) or nrem_scale == 0:
        nrem_scale = 1.0
    power_z = (envelope - nrem_med) / nrem_scale

    broad = butter_bandpass_filter(eeg_rs, target_fs, cfg.broad_band_low, cfg.broad_band_high)
    eeg_z = robust_z(eeg_rs)
    emg_z = robust_z(emg_rs) if emg_rs is not None else None

    candidate_events = find_hysteresis_events(power_z, cfg.thr_low_z, cfg.thr_high_z, target_fs, cfg.merge_gap_sec)

    rows = []
    margin_bands = [cfg.margin_low, cfg.margin_high]

    for s, e in candidate_events:
        if e <= s:
            continue
        dur = (e - s + 1) / target_fs
        if dur < cfg.min_duration_sec or dur > cfg.max_duration_sec:
            continue
        if np.mean(nrem_mask[s:e + 1]) < cfg.min_mask_fraction:
            continue
        if np.nanmax(power_z[s:e + 1]) > cfg.thr_max_z:
            continue
        if np.nanmax(np.abs(eeg_z[s:e + 1])) > cfg.artifact_eeg_robust_z:
            continue
        if emg_z is not None and np.nanmax(np.abs(emg_z[s:e + 1])) > cfg.artifact_emg_robust_z:
            # Conservative rejection of very movement-contaminated events.
            continue

        seg_broad = broad[s:e + 1]
        pos_peaks, _ = signal.find_peaks(seg_broad)
        neg_peaks, _ = signal.find_peaks(-seg_broad)
        n_cycles = len(pos_peaks)
        if n_cycles < cfg.min_cycles or n_cycles > cfg.max_cycles:
            continue

        # Spectral dominance: peak spindle-band power should exceed margin bands.
        pad = int(round(0.25 * target_fs))
        ps = max(0, s - pad)
        pe = min(n, e + 1 + pad)
        seg_for_psd = eeg_rs[ps:pe]
        cent_freq, spin_peak_power, spin_mean_power = psd_peak_frequency(
            seg_for_psd, target_fs, (cfg.broad_band_low, cfg.broad_band_high)
        )
        margin_peak_powers = []
        for band in margin_bands:
            _, p_peak, _ = psd_peak_frequency(seg_for_psd, target_fs, band)
            if np.isfinite(p_peak):
                margin_peak_powers.append(p_peak)
        if margin_peak_powers and np.isfinite(spin_peak_power):
            if spin_peak_power < max(margin_peak_powers):
                continue

        pos_amp = float(np.nanmax(seg_broad[pos_peaks])) if len(pos_peaks) else np.nan
        neg_amp = float(np.nanmin(seg_broad[neg_peaks])) if len(neg_peaks) else np.nan
        p2p = pos_amp - neg_amp if np.isfinite(pos_amp) and np.isfinite(neg_amp) else np.nan
        if len(pos_peaks):
            peak_loc = pos_peaks[int(np.nanargmax(seg_broad[pos_peaks]))]
            symmetry = peak_loc / max(1, len(seg_broad) - 1)
        else:
            symmetry = np.nan

        row = {
            "start_sec": s / target_fs,
            "end_sec": (e + 1) / target_fs,
            "duration_sec": dur,
            "peak_sigma_z": float(np.nanmax(power_z[s:e + 1])),
            "mean_sigma_z": float(np.nanmean(power_z[s:e + 1])),
            "sigma_auc_z_sec": float(np.trapz(np.maximum(power_z[s:e + 1], 0), dx=1 / target_fs)),
            "centFreq": cent_freq,
            "spin_peak_power": spin_peak_power,
            "spin_mean_power": spin_mean_power,
            "posPeak": pos_amp,
            "negPeak": neg_amp,
            "peak2peak": p2p,
            "noCycle": int(n_cycles),
            "symmetry": float(symmetry) if np.isfinite(symmetry) else np.nan,
            "mean_abs_broad_spindle": float(np.nanmean(np.abs(seg_broad))),
            "max_abs_broad_spindle": float(np.nanmax(np.abs(seg_broad))),
            "mean_abs_eeg_robust_z": float(np.nanmean(np.abs(eeg_z[s:e + 1]))),
            "max_abs_eeg_robust_z": float(np.nanmax(np.abs(eeg_z[s:e + 1]))),
            "mean_abs_emg_robust_z": float(np.nanmean(np.abs(emg_z[s:e + 1]))) if emg_z is not None else np.nan,
            "max_abs_emg_robust_z": float(np.nanmax(np.abs(emg_z[s:e + 1]))) if emg_z is not None else np.nan,
        }
        rows.append(row)

    params = {
        "target_fs": target_fs,
        "n_samples": n,
        "recording_sec": n / target_fs,
        "nrem_min": nrem_min,
        "n_candidates": len(candidate_events),
        "n_spindles": len(rows),
        "threshold_median_envelope_nrem": float(nrem_med),
        "threshold_scale_envelope_nrem": float(nrem_scale),
        **asdict(cfg),
    }
    return pd.DataFrame(rows), params


# -----------------------------------------------------------------------------
# Segment-level looping and summarization
# -----------------------------------------------------------------------------

def unique_segments_from_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "recording_name", "group", "week", "mouse_id", "segment_id",
        "file_path_raw_signals", "file_path_manual_state_annotation",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].drop_duplicates().copy()
    if "segment_id" in out.columns:
        out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce")
    if "mouse_id" in out.columns:
        out["mouse_id"] = pd.to_numeric(out["mouse_id"], errors="coerce")
    return out


def process_segment(row: pd.Series, cfg: SpindleConfig) -> tuple[pd.DataFrame, dict]:
    raw_path = Path(str(row["file_path_raw_signals"]))
    ann_path = Path(str(row["file_path_manual_state_annotation"]))
    eeg, emg, fs, eeg_label, emg_label = load_edf_eeg_emg(raw_path)
    recording_sec = len(eeg) / fs
    n_epochs = int(math.ceil(recording_sec / cfg.epoch_sec))
    states = load_stage_duration(ann_path, n_epochs=n_epochs, epoch_sec=cfg.epoch_sec)

    sp, params = detect_spindles_one_signal(eeg, fs, states, cfg, emg=emg)

    # Add segment metadata and context.
    meta = {
        "recording_name": row.get("recording_name", raw_path.parent.parent.name),
        "group": row.get("group", ""),
        "week": int(row["week"]) if pd.notna(row.get("week", np.nan)) else np.nan,
        "mouse_id": int(row["mouse_id"]) if pd.notna(row.get("mouse_id", np.nan)) else np.nan,
        "segment_id": int(row["segment_id"]) if pd.notna(row.get("segment_id", np.nan)) else np.nan,
        "file_path_raw_signals": str(raw_path),
        "file_path_manual_state_annotation": str(ann_path),
        "eeg_label": eeg_label,
        "emg_label": emg_label or "",
        "original_fs": fs,
        "recording_sec": recording_sec,
    }

    if len(sp):
        bouts = nrem_bouts_from_states(states, cfg.epoch_sec)
        context_rows = []
        for _, ev in sp.iterrows():
            mid = 0.5 * (float(ev["start_sec"]) + float(ev["end_sec"]))
            ctx = transition_context_for_time(states, mid, cfg.epoch_sec)
            bout_ctx = add_nrem_bout_context(ev.to_dict(), bouts)
            ctx.update(bout_ctx)
            for w in cfg.pre_transition_windows_sec:
                ctx[f"is_pre_transition_{w}s"] = bool(
                    np.isfinite(ctx.get("time_to_nrem_end_sec", np.nan))
                    and 0 <= ctx["time_to_nrem_end_sec"] <= w
                )
                ctx[f"is_pre_NREM_to_Wake_{w}s"] = bool(ctx[f"is_pre_transition_{w}s"] and ctx.get("next_state_after_NREM") == "Awake")
                ctx[f"is_pre_NREM_to_REM_{w}s"] = bool(ctx[f"is_pre_transition_{w}s"] and ctx.get("next_state_after_NREM") == "REM")
            context_rows.append(ctx)
        ctx_df = pd.DataFrame(context_rows)
        sp = pd.concat([sp.reset_index(drop=True), ctx_df.reset_index(drop=True)], axis=1)
        for k, v in meta.items():
            sp.insert(0, k, v)
        sp.insert(0, "spindle_id", [f"{meta['recording_name']}_seg{meta['segment_id']}_sp{i:05d}" for i in range(len(sp))])

    segment_metric = summarize_segment(meta, params, sp, states, cfg)
    return sp, segment_metric


def summarize_segment(meta: dict, params: dict, sp: pd.DataFrame, states: np.ndarray, cfg: SpindleConfig) -> dict:
    states_norm = np.array([normalize_state(s) for s in states])
    nrem_min = float(np.sum(states_norm == "NREM") * cfg.epoch_sec / 60.0)
    recording_min = float(len(states_norm) * cfg.epoch_sec / 60.0)
    n = len(sp)

    def mean_col(c: str) -> float:
        return float(pd.to_numeric(sp[c], errors="coerce").mean()) if n and c in sp.columns else np.nan

    def median_col(c: str) -> float:
        return float(pd.to_numeric(sp[c], errors="coerce").median()) if n and c in sp.columns else np.nan

    out = {
        **meta,
        "recording_min": recording_min,
        "NREM_min": nrem_min,
        "n_spindles": int(n),
        "spindle_density_per_NREM_min": n / nrem_min if nrem_min > 0 else np.nan,
        "mean_spindle_duration_sec": mean_col("duration_sec"),
        "median_spindle_duration_sec": median_col("duration_sec"),
        "mean_spindle_peak_sigma_z": mean_col("peak_sigma_z"),
        "mean_spindle_sigma_auc_z_sec": mean_col("sigma_auc_z_sec"),
        "mean_spindle_centFreq": mean_col("centFreq"),
        "mean_spindle_peak2peak": mean_col("peak2peak"),
        "mean_spindle_cycles": mean_col("noCycle"),
        "mean_spindle_symmetry": mean_col("symmetry"),
        "detector_n_candidates": int(params.get("n_candidates", 0)),
        "detector_threshold_scale_envelope_nrem": params.get("threshold_scale_envelope_nrem", np.nan),
        "detector_threshold_median_envelope_nrem": params.get("threshold_median_envelope_nrem", np.nan),
    }

    for w in cfg.pre_transition_windows_sec:
        col = f"is_pre_transition_{w}s"
        wake_col = f"is_pre_NREM_to_Wake_{w}s"
        rem_col = f"is_pre_NREM_to_REM_{w}s"
        n_pre = int(sp[col].sum()) if n and col in sp.columns else 0
        n_pre_wake = int(sp[wake_col].sum()) if n and wake_col in sp.columns else 0
        n_pre_rem = int(sp[rem_col].sum()) if n and rem_col in sp.columns else 0
        out[f"n_spindles_pre_transition_{w}s"] = n_pre
        out[f"fraction_spindles_pre_transition_{w}s"] = n_pre / n if n else np.nan
        out[f"spindle_density_pre_transition_{w}s_per_NREM_min"] = n_pre / nrem_min if nrem_min > 0 else np.nan
        out[f"n_spindles_pre_NREM_to_Wake_{w}s"] = n_pre_wake
        out[f"spindle_density_pre_NREM_to_Wake_{w}s_per_NREM_min"] = n_pre_wake / nrem_min if nrem_min > 0 else np.nan
        out[f"n_spindles_pre_NREM_to_REM_{w}s"] = n_pre_rem
        out[f"spindle_density_pre_NREM_to_REM_{w}s_per_NREM_min"] = n_pre_rem / nrem_min if nrem_min > 0 else np.nan

    return out


def summarize_mouse_week(segment_metrics: pd.DataFrame) -> pd.DataFrame:
    if len(segment_metrics) == 0:
        return pd.DataFrame()
    key = ["group", "week", "mouse_id"]
    seg = segment_metrics.copy()

    # Weighted means helper by spindle counts or NREM duration where appropriate.
    rows = []
    for k, g in seg.groupby(key, dropna=False):
        row = dict(zip(key, k))
        row["recording_min"] = pd.to_numeric(g["recording_min"], errors="coerce").sum()
        row["NREM_min"] = pd.to_numeric(g["NREM_min"], errors="coerce").sum()
        row["n_spindles"] = int(pd.to_numeric(g["n_spindles"], errors="coerce").fillna(0).sum())
        row["n_segments_with_spindle_detection"] = int(len(g))
        row["spindle_density_per_NREM_min"] = row["n_spindles"] / row["NREM_min"] if row["NREM_min"] > 0 else np.nan

        # For event features, weighted by segment spindle count.
        weights = pd.to_numeric(g["n_spindles"], errors="coerce").fillna(0).values.astype(float)
        for c in [
            "mean_spindle_duration_sec",
            "mean_spindle_peak_sigma_z",
            "mean_spindle_sigma_auc_z_sec",
            "mean_spindle_centFreq",
            "mean_spindle_peak2peak",
            "mean_spindle_cycles",
            "mean_spindle_symmetry",
        ]:
            vals = pd.to_numeric(g[c], errors="coerce").values.astype(float) if c in g.columns else np.array([])
            good = np.isfinite(vals) & (weights > 0)
            row[c] = float(np.average(vals[good], weights=weights[good])) if np.any(good) else np.nan

        for c in g.columns:
            if c.startswith("n_spindles_pre_"):
                row[c] = int(pd.to_numeric(g[c], errors="coerce").fillna(0).sum())
            if c.startswith("spindle_density_pre_") and c.endswith("_per_NREM_min"):
                n_col = c.replace("spindle_density_", "n_spindles_").replace("_per_NREM_min", "")
                if n_col in row:
                    row[c] = row[n_col] / row["NREM_min"] if row["NREM_min"] > 0 else np.nan

        # Fractions after summed counts.
        for c in list(row.keys()):
            if c.startswith("n_spindles_pre_transition_"):
                tag = c.replace("n_spindles_pre_transition_", "")
                row[f"fraction_spindles_pre_transition_{tag}"] = row[c] / row["n_spindles"] if row["n_spindles"] else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(key)


def merge_with_rbd_metrics(spindle_mouse_week: pd.DataFrame, rbd_metrics_path: str | Path) -> pd.DataFrame:
    rbd = pd.read_csv(rbd_metrics_path)
    sp = spindle_mouse_week.copy()
    for df in [rbd, sp]:
        for c in ["week", "mouse_id"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        if "group" in df.columns:
            df["group"] = df["group"].astype(str)
    merged = rbd.merge(sp, on=["group", "week", "mouse_id"], how="left", suffixes=("", "_spindle"))
    return merged


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = SpindleConfig(
        target_fs=args.target_fs,
        epoch_sec=args.epoch_sec,
        sigma_band_low=args.sigma_low,
        sigma_band_high=args.sigma_high,
        broad_band_low=args.broad_low,
        broad_band_high=args.broad_high,
        thr_low_z=args.thr_low_z,
        thr_high_z=args.thr_high_z,
        thr_max_z=args.thr_max_z,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        min_mask_fraction=args.min_mask_fraction,
        artifact_emg_robust_z=args.artifact_emg_z,
        artifact_eeg_robust_z=args.artifact_eeg_z,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spindle_detector_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    source = pd.read_csv(args.segments_csv)
    segments = unique_segments_from_table(source)
    if args.limit_segments is not None:
        segments = segments.head(args.limit_segments).copy()
    if args.only_group:
        segments = segments[segments["group"].astype(str).isin(args.only_group)].copy()
    if args.only_week:
        weeks = set(int(w) for w in args.only_week)
        segments = segments[pd.to_numeric(segments["week"], errors="coerce").isin(weeks)].copy()

    print(f"Segments to process: {len(segments)}")

    all_events = []
    seg_metrics = []
    errors = []

    for i, row in segments.reset_index(drop=True).iterrows():
        label = f"{row.get('group','')} W{row.get('week','')} M{row.get('mouse_id','')} seg{row.get('segment_id','')}"
        print(f"[{i+1}/{len(segments)}] {label}")
        try:
            sp, sm = process_segment(row, cfg)
            if len(sp):
                all_events.append(sp)
            seg_metrics.append(sm)
            print(f"    spindles: {len(sp)}")
        except Exception as e:
            print(f"    ERROR: {repr(e)}")
            errors.append({
                "recording_name": row.get("recording_name", ""),
                "group": row.get("group", ""),
                "week": row.get("week", ""),
                "mouse_id": row.get("mouse_id", ""),
                "segment_id": row.get("segment_id", ""),
                "file_path_raw_signals": row.get("file_path_raw_signals", ""),
                "error": repr(e),
            })

    spindle_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    segment_metrics = pd.DataFrame(seg_metrics)
    mouse_week = summarize_mouse_week(segment_metrics)

    spindle_events.to_csv(out_dir / "spindle_events.csv", index=False)
    segment_metrics.to_csv(out_dir / "spindle_segment_metrics.csv", index=False)
    mouse_week.to_csv(out_dir / "spindle_mouse_week_metrics.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "spindle_detection_errors.csv", index=False)

    print("\nSaved:")
    print(out_dir / "spindle_events.csv")
    print(out_dir / "spindle_segment_metrics.csv")
    print(out_dir / "spindle_mouse_week_metrics.csv")
    print(out_dir / "spindle_detection_errors.csv")

    if args.merge_rbd_metrics:
        merged = merge_with_rbd_metrics(mouse_week, args.merge_rbd_metrics)
        merged.to_csv(out_dir / "mouse_week_metrics_with_spindles.csv", index=False)
        print(out_dir / "mouse_week_metrics_with_spindles.csv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect NREM spindles and summarize spindle metrics.")
    p.add_argument("--segments-csv", required=True, help="Event table or segment manifest CSV with EDF and annotation paths.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--merge-rbd-metrics", default="", help="Optional live_mouse_level_EMG_REM_metrics.csv to merge with spindle metrics.")
    p.add_argument("--target-fs", type=float, default=200.0)
    p.add_argument("--epoch-sec", type=float, default=5.0)
    p.add_argument("--sigma-low", type=float, default=10.0)
    p.add_argument("--sigma-high", type=float, default=14.0)
    p.add_argument("--broad-low", type=float, default=9.0)
    p.add_argument("--broad-high", type=float, default=16.0)
    p.add_argument("--thr-low-z", type=float, default=1.0)
    p.add_argument("--thr-high-z", type=float, default=3.0)
    p.add_argument("--thr-max-z", type=float, default=20.0)
    p.add_argument("--min-duration", type=float, default=0.40)
    p.add_argument("--max-duration", type=float, default=2.00)
    p.add_argument("--min-mask-fraction", type=float, default=0.80)
    p.add_argument("--artifact-eeg-z", type=float, default=10.0)
    p.add_argument("--artifact-emg-z", type=float, default=8.0)
    p.add_argument("--limit-segments", type=int, default=None, help="Debug option: process only first N segments.")
    p.add_argument("--only-group", nargs="*", default=None, help="Optional group filter, e.g. --only-group WT PD")
    p.add_argument("--only-week", nargs="*", default=None, help="Optional week filter, e.g. --only-week 2 21")
    return p


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
