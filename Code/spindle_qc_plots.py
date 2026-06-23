#!/usr/bin/env python3
"""
Generate spindle-detection QC plots for the Katia PD/RBD project.

This script is intentionally offline/batch. It reads spindle_events.csv from
spindle_pipeline.py, loads short EEG/EMG windows around selected spindles, and
saves PNG plots that can be inspected directly or displayed inside Streamlit.

Typical use from Project_PD_RBD_Katia folder:

python Code/spindle_qc_plots.py \
  --spindle-events Data/prepared_data/manifests/spindle_detection/spindle_events.csv \
  --config Data/prepared_data/manifests/spindle_detection/spindle_detector_config.json \
  --out-dir Data/prepared_data/manifests/spindle_detection/qc_plots \
  --n-random 30 \
  --n-low-confidence 20 \
  --n-high-confidence 20 \
  --window-sec 8

Outputs:
    qc_spindle_*.png
    spindle_qc_plot_index.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

try:
    from pyedflib import EdfReader
except Exception:  # pragma: no cover
    EdfReader = None


STATE_COLORS = {
    "Awake": 0,
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Undefined": 3,
}

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


def normalize_state(x: object) -> str:
    return STATE_NORMALIZATION.get(str(x).strip(), str(x).strip())


def robust_z(x: np.ndarray, clip: Optional[float] = 8.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    z = (x - med) / scale
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z


def safe_filename(x: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))
    return s[:max_len]


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
        eeg_idx = 0
    eeg_label = labels[eeg_idx]
    emg_label = labels[emg_idx] if emg_idx is not None else None
    return eeg_idx, emg_idx, eeg_label, emg_label


def read_edf_window(edf_path: str | Path, start_s: float, end_s: float):
    if EdfReader is None:
        raise ImportError("pyedflib is not installed. Install with: pip install pyedflib")
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    with EdfReader(str(edf_path)) as reader:
        labels = reader.getSignalLabels()
        eeg_idx, emg_idx, eeg_label, emg_label = infer_channel_indices(labels)
        fs = float(reader.getSampleFrequency(eeg_idx))
        n_total = reader.getNSamples()[eeg_idx]
        recording_sec = n_total / fs

        start_s = max(0.0, float(start_s))
        end_s = min(float(end_s), recording_sec)
        start_sample = int(round(start_s * fs))
        end_sample = max(start_sample + 1, int(round(end_s * fs)))
        n_samples = end_sample - start_sample

        eeg = np.asarray(reader.readSignal(eeg_idx, start_sample, n_samples), dtype=float)
        emg = None
        if emg_idx is not None:
            emg_fs = float(reader.getSampleFrequency(emg_idx))
            emg_start = int(round(start_s * emg_fs))
            emg_end = max(emg_start + 1, int(round(end_s * emg_fs)))
            emg = np.asarray(reader.readSignal(emg_idx, emg_start, emg_end - emg_start), dtype=float)
            if abs(emg_fs - fs) > 1e-6:
                emg = signal.resample(emg, len(eeg))

        t = np.arange(len(eeg), dtype=float) / fs + start_s
        return t, eeg, emg, fs, eeg_label, emg_label


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.1, float(low))
    high = min(float(high), nyq * 0.95)
    if high <= low:
        return np.asarray(x, dtype=float)
    sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    return signal.sosfiltfilt(sos, x)


def smooth_moving_average(x: np.ndarray, fs: float, sec: float) -> np.ndarray:
    win = max(1, int(round(float(sec) * fs)))
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def load_stage_duration(path: str | Path, epoch_sec: float = 5.0):
    path = Path(path)
    if not path.exists():
        return np.array([], dtype=object)
    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    states: list[str] = []
    if not lines:
        return np.array([], dtype=object)
    if lines[0].startswith("*Duration"):
        prev_end_sec = 0.0
        for line in lines[2:]:
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue
            label = " ".join(parts[:-1])
            try:
                end_sec = float(parts[-1])
            except Exception:
                continue
            start_epoch = int(round(prev_end_sec / epoch_sec))
            end_epoch = int(round(end_sec / epoch_sec))
            states.extend([normalize_state(label)] * max(0, end_epoch - start_epoch))
            prev_end_sec = end_sec
    else:
        states = [normalize_state(x) for x in lines]
    return np.asarray(states, dtype=object)


def plot_state_bar(ax, annotation_path: str | Path, start_s: float, end_s: float, epoch_sec: float = 5.0):
    states = load_stage_duration(annotation_path, epoch_sec=epoch_sec)
    if len(states) == 0:
        ax.text(0.5, 0.5, "No hypnogram", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    e0 = max(0, int(np.floor(start_s / epoch_sec)))
    e1 = min(len(states), int(np.ceil(end_s / epoch_sec)))
    if e1 <= e0:
        ax.text(0.5, 0.5, "Hypnogram outside window", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    codes = np.array([STATE_COLORS.get(normalize_state(s), 3) for s in states[e0:e1]])
    extent = [e0 * epoch_sec, e1 * epoch_sec, 0, 1]
    cmap = plt.matplotlib.colors.ListedColormap(["tab:blue", "tab:orange", "tab:green", "lightgray"])
    ax.imshow(codes.reshape(1, -1), aspect="auto", interpolation="nearest", extent=extent, cmap=cmap, vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_ylabel("State")
    ax.set_xlim(start_s, end_s)


def choose_spindles(df: pd.DataFrame, n_random: int, n_low: int, n_high: int, n_near_transition: int, seed: int) -> pd.DataFrame:
    pieces = []
    rng = np.random.default_rng(seed)
    if len(df) == 0:
        return df
    if n_random > 0:
        pieces.append(df.sample(n=min(n_random, len(df)), random_state=seed))
    if n_low > 0 and "peak_sigma_z" in df.columns:
        pieces.append(df.sort_values("peak_sigma_z", ascending=True).head(n_low))
    if n_high > 0 and "peak_sigma_z" in df.columns:
        pieces.append(df.sort_values("peak_sigma_z", ascending=False).head(n_high))
    if n_near_transition > 0 and "distance_to_nearest_transition_sec" in df.columns:
        tmp = df.copy()
        tmp["_dist"] = pd.to_numeric(tmp["distance_to_nearest_transition_sec"], errors="coerce")
        pieces.append(tmp.sort_values("_dist", ascending=True).head(n_near_transition).drop(columns=["_dist"], errors="ignore"))
    out = pd.concat(pieces, ignore_index=True) if pieces else df.head(0)
    out = out.drop_duplicates("spindle_id", keep="first") if "spindle_id" in out.columns else out.drop_duplicates()
    # Shuffle for easier manual inspection.
    if len(out):
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def make_spindle_qc_plot(row: pd.Series, all_events: pd.DataFrame, config: dict, out_path: Path, window_sec: float = 8.0):
    start = float(row["start_sec"])
    end = float(row["end_sec"])
    mid = 0.5 * (start + end)
    plot_start = max(0.0, mid - window_sec / 2.0)
    plot_end = mid + window_sec / 2.0

    t, eeg, emg, fs, eeg_label, emg_label = read_edf_window(row["file_path_raw_signals"], plot_start, plot_end)

    broad = bandpass(eeg, fs, config.get("broad_band_low", 9.0), config.get("broad_band_high", 16.0))
    sigma = bandpass(eeg, fs, config.get("sigma_band_low", 10.0), config.get("sigma_band_high", 14.0))
    env = np.abs(signal.hilbert(sigma))
    env = smooth_moving_average(env, fs, config.get("envelope_smooth_sec", 0.2))
    env_z = robust_z(env, clip=12)

    rows = 5 if emg is not None else 4
    heights = [0.35, 1.0, 1.0, 1.0] + ([0.8] if emg is not None else [])
    fig, axes = plt.subplots(rows, 1, figsize=(12.5, 7.5 if emg is not None else 6.5), sharex=True, gridspec_kw={"height_ratios": heights})
    if rows == 1:
        axes = [axes]

    ax_i = 0
    ann_path = row.get("file_path_manual_state_annotation", "")
    if isinstance(ann_path, str) and ann_path:
        plot_state_bar(axes[ax_i], ann_path, plot_start, plot_end, epoch_sec=config.get("epoch_sec", 5.0))
    else:
        axes[ax_i].text(0.5, 0.5, "No hypnogram path", ha="center", va="center", transform=axes[ax_i].transAxes)
        axes[ax_i].set_axis_off()
    ax_i += 1

    axes[ax_i].plot(t, robust_z(eeg, clip=8), linewidth=0.55)
    axes[ax_i].set_ylabel(f"EEG\n{eeg_label}\nrobust z")
    ax_i += 1

    axes[ax_i].plot(t, robust_z(broad, clip=8), linewidth=0.65)
    axes[ax_i].set_ylabel(f"{config.get('broad_band_low',9):.0f}-{config.get('broad_band_high',16):.0f} Hz\nrobust z")
    ax_i += 1

    axes[ax_i].plot(t, env_z, linewidth=0.9, label="Sigma envelope, local robust z")
    axes[ax_i].axhline(config.get("thr_low_z", 1.0), linestyle="--", linewidth=0.8, color="gray", label="low threshold")
    axes[ax_i].axhline(config.get("thr_high_z", 3.0), linestyle="--", linewidth=0.8, color="black", label="high threshold")
    axes[ax_i].set_ylabel(f"{config.get('sigma_band_low',10):.0f}-{config.get('sigma_band_high',14):.0f} Hz\nenvelope z")
    axes[ax_i].legend(loc="upper right", fontsize=7)
    ax_i += 1

    if emg is not None:
        axes[ax_i].plot(t, robust_z(emg, clip=8), linewidth=0.45)
        axes[ax_i].set_ylabel(f"EMG\n{emg_label}\nrobust z")
        ax_i += 1

    # Mark all spindles in the visible window for same recording/segment.
    same = all_events.copy()
    for col in ["group", "recording_name"]:
        if col in same.columns and col in row.index:
            same = same[same[col].astype(str) == str(row[col])]
    for col in ["week", "mouse_id", "segment_id"]:
        if col in same.columns and col in row.index:
            same = same[pd.to_numeric(same[col], errors="coerce") == float(row[col])]
    same = same[(pd.to_numeric(same["end_sec"], errors="coerce") >= plot_start) & (pd.to_numeric(same["start_sec"], errors="coerce") <= plot_end)]

    selected_id = str(row.get("spindle_id", ""))
    for ax in axes:
        for _, r in same.iterrows():
            s = float(r["start_sec"]); e = float(r["end_sec"])
            is_selected = str(r.get("spindle_id", "")) == selected_id
            ax.axvspan(s, e, color="red" if is_selected else "gold", alpha=0.28 if is_selected else 0.18)
        ax.axvline(start, linestyle=":", color="red", linewidth=1.0)
        ax.axvline(end, linestyle=":", color="red", linewidth=1.0)
        ax.set_xlim(plot_start, plot_end)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time in recording (s)")

    title = (
        f"Spindle QC: {row.get('spindle_id', '')}\n"
        f"{row.get('group','')} W{row.get('week','')} mouse {row.get('mouse_id','')} seg {row.get('segment_id','')} | "
        f"duration={float(row.get('duration_sec', np.nan)):.2f}s, "
        f"peak_sigma_z={float(row.get('peak_sigma_z', np.nan)):.2f}, "
        f"centFreq={float(row.get('centFreq', np.nan)):.2f}Hz, "
        f"state={row.get('state_at_event','')}, next={row.get('next_state_after_NREM','')}"
    )
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spindle-events", required=True, help="Path to spindle_events.csv")
    ap.add_argument("--config", default="", help="Path to spindle_detector_config.json")
    ap.add_argument("--out-dir", required=True, help="Output folder for QC PNGs")
    ap.add_argument("--window-sec", type=float, default=8.0, help="Seconds shown around each spindle midpoint")
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-low-confidence", type=int, default=20, help="Lowest peak_sigma_z examples")
    ap.add_argument("--n-high-confidence", type=int, default=20, help="Highest peak_sigma_z examples")
    ap.add_argument("--n-near-transition", type=int, default=20, help="Closest-to-transition examples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    events_path = Path(args.spindle_events)
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    events = pd.read_csv(events_path)
    if len(events) == 0:
        raise ValueError("spindle_events.csv is empty")

    config = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())
    else:
        # Defaults matching spindle_pipeline.py.
        config = {
            "target_fs": 200.0,
            "epoch_sec": 5.0,
            "sigma_band_low": 10.0,
            "sigma_band_high": 14.0,
            "broad_band_low": 9.0,
            "broad_band_high": 16.0,
            "envelope_smooth_sec": 0.2,
            "thr_low_z": 1.0,
            "thr_high_z": 3.0,
        }

    chosen = choose_spindles(
        events,
        n_random=args.n_random,
        n_low=args.n_low_confidence,
        n_high=args.n_high_confidence,
        n_near_transition=args.n_near_transition,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []

    for i, (_, row) in enumerate(chosen.iterrows(), start=1):
        spindle_id = str(row.get("spindle_id", f"spindle_{i:05d}"))
        png_name = f"qc_spindle_{i:04d}_{safe_filename(spindle_id)}.png"
        png_path = out_dir / png_name
        try:
            make_spindle_qc_plot(row, events, config, png_path, window_sec=args.window_sec)
            rec = row.to_dict()
            rec.update({
                "qc_plot_path": str(png_path),
                "qc_plot_file": png_name,
                "qc_plot_order": i,
            })
            rows.append(rec)
            print(f"[{i}/{len(chosen)}] saved {png_path}")
        except Exception as e:
            errors.append({"spindle_id": spindle_id, "error": repr(e)})
            print(f"[{i}/{len(chosen)}] ERROR {spindle_id}: {repr(e)}")

    index = pd.DataFrame(rows)
    index_path = out_dir / "spindle_qc_plot_index.csv"
    index.to_csv(index_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "spindle_qc_plot_errors.csv", index=False)

    print("\nDone.")
    print(f"QC plots: {out_dir}")
    print(f"Index: {index_path}")
    if errors:
        print(f"Errors: {out_dir / 'spindle_qc_plot_errors.csv'}")


if __name__ == "__main__":
    main()
