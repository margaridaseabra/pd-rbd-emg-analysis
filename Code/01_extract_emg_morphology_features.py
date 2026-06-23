from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    from pyedflib import EdfReader
except Exception:
    EdfReader = None


BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "PD-Katia/Data/prepared_data/manifests"
)

DEFAULT_EVENTS = (
    BASE
    / "EMG_burst_detection_NREM_baseline"
    / "qc_ready"
    / "EMG_episodes_NREMbaseline_qc_ready.csv"
)

DEFAULT_OUT_DIR = BASE / "EMG_unsupervised_morphology"

VIDEO_QC_OUT = (
    BASE
    / "EMG_burst_detection_NREM_baseline"
    / "qc_ready"
    / "interactive_QC_annotations.csv"
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def infer_emg_index(labels):
    for i, lab in enumerate(labels):
        if "EMG" in str(lab).upper():
            return i
    raise ValueError(f"Could not infer EMG channel from EDF labels: {labels}")


def read_emg_window(edf_path, start_s, end_s):
    if EdfReader is None:
        raise ImportError("pyedflib is not installed. Install with: pip install pyedflib")

    edf_path = Path(edf_path)

    with EdfReader(str(edf_path)) as reader:
        labels = reader.getSignalLabels()
        emg_idx = infer_emg_index(labels)
        fs = float(reader.getSampleFrequency(emg_idx))
        n_total = int(reader.getNSamples()[emg_idx])

        start_sample = max(0, int(round(start_s * fs)))
        end_sample = min(n_total, max(start_sample + 1, int(round(end_s * fs))))
        n_samples = end_sample - start_sample

        emg = reader.readSignal(emg_idx, start_sample, n_samples)
        t = np.arange(len(emg)) / fs + start_sample / fs

    return t, np.asarray(emg, dtype=float), fs, labels[emg_idx]


def get_edf_duration_sec(edf_path):
    if EdfReader is None:
        return np.nan

    try:
        with EdfReader(str(edf_path)) as reader:
            labels = reader.getSignalLabels()
            emg_idx = infer_emg_index(labels)
            fs = float(reader.getSampleFrequency(emg_idx))
            n_total = int(reader.getNSamples()[emg_idx])
            return n_total / fs
    except Exception:
        return np.nan


def highpass_emg(emg, fs, hp_hz=10.0):
    emg = np.asarray(emg, dtype=float)
    emg = emg - np.nanmedian(emg)

    if fs <= 2 * hp_hz or len(emg) < int(fs):
        return emg

    try:
        sos = signal.butter(4, hp_hz, btype="highpass", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, emg)
    except Exception:
        return emg


def moving_rms(x, fs, window_sec=0.25):
    x = np.asarray(x, dtype=float)
    win = max(1, int(round(window_sec * fs)))
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(x ** 2, kernel, mode="same"))


def robust_scale_from_baseline(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return 0.0, 1.0

    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(x))

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return med, scale


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    return float(np.nanpercentile(x, q)) if np.isfinite(x).any() else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


def safe_skew(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.nanstd(x) == 0:
        return np.nan
    return float(stats.skew(x))


def safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.nanstd(x) == 0:
        return np.nan
    return float(stats.kurtosis(x))


def find_active_segments(active, fs, min_dur_sec=0.05, merge_gap_sec=0.10):
    active = np.asarray(active, dtype=bool)

    if len(active) == 0 or not active.any():
        return []

    starts = np.where(active & np.r_[True, ~active[:-1]])[0]
    ends = np.where(active & np.r_[~active[1:], True])[0] + 1

    segments = [(int(s), int(e)) for s, e in zip(starts, ends)]

    # Merge short gaps.
    merged = []
    max_gap = int(round(merge_gap_sec * fs))

    for s, e in segments:
        if not merged:
            merged.append([s, e])
            continue

        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_len = int(round(min_dur_sec * fs))
    merged = [(s, e) for s, e in merged if (e - s) >= min_len]

    return merged


def spectral_features(x, fs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    out = {
        "emg_spectral_centroid_hz": np.nan,
        "emg_spectral_median_hz": np.nan,
        "emg_spectral_bandpower_10_50": np.nan,
        "emg_spectral_bandpower_50_150": np.nan,
        "emg_spectral_high_fraction_50_150": np.nan,
    }

    if len(x) < max(32, int(0.25 * fs)):
        return out

    try:
        nperseg = min(len(x), int(round(fs)))
        f, pxx = signal.welch(x, fs=fs, nperseg=nperseg)

        keep = (f >= 10) & (f <= min(250, fs / 2))
        f = f[keep]
        pxx = pxx[keep]

        if len(f) == 0 or np.nansum(pxx) <= 0:
            return out

        total = np.nansum(pxx)
        centroid = np.nansum(f * pxx) / total

        cs = np.cumsum(pxx)
        median_freq = f[np.searchsorted(cs, total / 2)]

        bp_10_50 = np.nansum(pxx[(f >= 10) & (f < 50)])
        bp_50_150 = np.nansum(pxx[(f >= 50) & (f < 150)])

        out.update({
            "emg_spectral_centroid_hz": float(centroid),
            "emg_spectral_median_hz": float(median_freq),
            "emg_spectral_bandpower_10_50": float(bp_10_50),
            "emg_spectral_bandpower_50_150": float(bp_50_150),
            "emg_spectral_high_fraction_50_150": float(bp_50_150 / total),
        })

    except Exception:
        pass

    return out


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_one_event_features(
    row,
    pre_s=20.0,
    post_s=20.0,
    rms_window_sec=0.25,
    active_z_threshold=3.0,
):
    start_s = float(row["start_sec"])
    end_s = float(row["end_sec"])
    duration_s = max(1e-6, end_s - start_s)

    raw_path = Path(row["file_path_raw_signals"])

    read_start = max(0.0, start_s - pre_s)
    read_end = end_s + post_s

    t, emg_raw, fs, emg_label = read_emg_window(raw_path, read_start, read_end)

    emg_hp = highpass_emg(emg_raw, fs)
    emg_abs = np.abs(emg_hp)
    emg_rms = moving_rms(emg_hp, fs, window_sec=rms_window_sec)

    event_mask = (t >= start_s) & (t <= end_s)
    pre_mask = (t >= max(0, start_s - pre_s)) & (t < start_s)
    post_mask = (t > end_s) & (t <= end_s + post_s)
    baseline_mask = pre_mask | post_mask

    if event_mask.sum() < 2:
        raise ValueError("Event window contains too few samples.")

    event_raw = emg_hp[event_mask]
    event_abs = emg_abs[event_mask]
    event_rms = emg_rms[event_mask]

    baseline_rms = emg_rms[baseline_mask]
    baseline_abs = emg_abs[baseline_mask]

    if len(baseline_rms) < int(1.0 * fs):
        baseline_rms = emg_rms
        baseline_abs = emg_abs

    base_rms_med, base_rms_scale = robust_scale_from_baseline(baseline_rms)
    base_abs_med, base_abs_scale = robust_scale_from_baseline(baseline_abs)

    event_rms_z = (event_rms - base_rms_med) / base_rms_scale
    event_abs_z = (event_abs - base_abs_med) / base_abs_scale

    active_threshold = base_rms_med + active_z_threshold * base_rms_scale
    active_event = event_rms > active_threshold

    active_segments = find_active_segments(
        active_event,
        fs=fs,
        min_dur_sec=0.05,
        merge_gap_sec=0.10,
    )

    subburst_durations = np.array([(e - s) / fs for s, e in active_segments], dtype=float)
    n_subbursts = int(len(active_segments))
    total_active_duration = float(np.nansum(subburst_durations)) if n_subbursts else 0.0

    # Peak / burstiness / regularity.
    try:
        min_peak_distance = max(1, int(round(0.05 * fs)))
        peaks, peak_props = signal.find_peaks(
            event_rms,
            height=active_threshold,
            distance=min_peak_distance,
        )
    except Exception:
        peaks = np.array([], dtype=int)

    if len(peaks) >= 2:
        peak_intervals = np.diff(peaks) / fs
        peak_interval_mean = safe_mean(peak_intervals)
        peak_interval_std = safe_std(peak_intervals)
        peak_interval_cv = (
            peak_interval_std / peak_interval_mean
            if np.isfinite(peak_interval_mean) and peak_interval_mean > 0
            else np.nan
        )
        peak_regular_score = 1.0 / (1.0 + peak_interval_cv) if np.isfinite(peak_interval_cv) else np.nan
    else:
        peak_interval_mean = np.nan
        peak_interval_std = np.nan
        peak_interval_cv = np.nan
        peak_regular_score = np.nan

    # Shape.
    peak_idx = int(np.nanargmax(event_rms)) if len(event_rms) else 0
    peak_time_frac = peak_idx / max(1, len(event_rms) - 1)

    if active_event.any():
        active_idx = np.where(active_event)[0]
        onset_idx = int(active_idx[0])
        offset_idx = int(active_idx[-1])
    else:
        onset_idx = 0
        offset_idx = len(event_rms) - 1

    rise_time_s = max(0.0, (peak_idx - onset_idx) / fs)
    decay_time_s = max(0.0, (offset_idx - peak_idx) / fs)
    rise_decay_ratio = rise_time_s / decay_time_s if decay_time_s > 0 else np.nan

    event_rms_mean = safe_mean(event_rms)
    event_rms_max = safe_percentile(event_rms, 100)
    event_rms_p95 = safe_percentile(event_rms, 95)

    peak_to_mean_ratio = (
        event_rms_max / event_rms_mean
        if np.isfinite(event_rms_mean) and event_rms_mean > 0
        else np.nan
    )

    # Background tone around the episode.
    pre_rms = emg_rms[pre_mask]
    post_rms = emg_rms[post_mask]

    features = {
        # identifiers
        "qc_event_id": row.get("qc_event_id", np.nan),
        "stable_event_key": row.get("stable_event_key", ""),
        "recording_name": row.get("recording_name", ""),
        "group": row.get("group", ""),
        "week": row.get("week", np.nan),
        "mouse_id": row.get("mouse_id", np.nan),
        "segment_id": row.get("segment_id", np.nan),
        "primary_category": row.get("primary_category", ""),
        "event_class": row.get("event_class", ""),
        "manual_state_center": row.get("manual_state_center", ""),
        "EEGonly_state_center": row.get("EEGonly_state_center", ""),
        "full_state_center": row.get("full_state_center", ""),
        "start_sec": start_s,
        "end_sec": end_s,
        "duration_sec": duration_s,
        "file_path_raw_signals": str(raw_path),
        "emg_label": str(emg_label),
        "fs": fs,

        # existing model/probability columns if present
        "P_REM_EEGonly": row.get("P_REM_EEGonly", np.nan),
        "P_REM_FULL": row.get("P_REM_FULL", np.nan),
        "P_Wake_EEGonly": row.get("P_Wake_EEGonly", np.nan),
        "P_Wake_FULL": row.get("P_Wake_FULL", np.nan),
        "delta_REM": row.get("delta_REM", np.nan),
        "max_EMG_z_existing": row.get("max_EMG_z", np.nan),
        "distance_to_transition_sec": row.get("distance_to_transition_sec", np.nan),

        # duration
        "morph_duration_sec": duration_s,

        # amplitude
        "emg_abs_mean": safe_mean(event_abs),
        "emg_abs_max": safe_percentile(event_abs, 100),
        "emg_abs_p95": safe_percentile(event_abs, 95),
        "emg_abs_z_mean": safe_mean(event_abs_z),
        "emg_abs_z_max": safe_percentile(event_abs_z, 100),
        "emg_abs_z_p95": safe_percentile(event_abs_z, 95),

        "emg_rms_mean": event_rms_mean,
        "emg_rms_max": event_rms_max,
        "emg_rms_p95": event_rms_p95,
        "emg_rms_auc": float(np.nansum(event_rms) / fs),
        "emg_rms_z_mean": safe_mean(event_rms_z),
        "emg_rms_z_max": safe_percentile(event_rms_z, 100),
        "emg_rms_z_p95": safe_percentile(event_rms_z, 95),

        "emg_event_to_background_rms_ratio": (
            event_rms_mean / base_rms_med if np.isfinite(base_rms_med) and base_rms_med > 0 else np.nan
        ),

        # background tone
        "background_rms_median": base_rms_med,
        "background_rms_scale": base_rms_scale,
        "pre_event_rms_median": safe_median(pre_rms),
        "post_event_rms_median": safe_median(post_rms),
        "pre_post_tone_change": safe_median(post_rms) - safe_median(pre_rms),

        # burstiness / duty cycle
        "n_subbursts": n_subbursts,
        "subburst_rate_per_sec": n_subbursts / duration_s,
        "total_active_duration_sec": total_active_duration,
        "active_fraction_in_event": total_active_duration / duration_s,
        "mean_subburst_duration_sec": safe_mean(subburst_durations) if n_subbursts else np.nan,
        "max_subburst_duration_sec": safe_percentile(subburst_durations, 100) if n_subbursts else np.nan,

        # peakiness / regularity
        "n_rms_peaks": int(len(peaks)),
        "rms_peak_rate_per_sec": len(peaks) / duration_s,
        "rms_peak_interval_mean_sec": peak_interval_mean,
        "rms_peak_interval_std_sec": peak_interval_std,
        "rms_peak_interval_cv": peak_interval_cv,
        "rms_peak_regularity_score": peak_regular_score,

        # shape
        "rms_peak_time_fraction": peak_time_frac,
        "rms_rise_time_sec": rise_time_s,
        "rms_decay_time_sec": decay_time_s,
        "rms_rise_decay_ratio": rise_decay_ratio,
        "rms_peak_to_mean_ratio": peak_to_mean_ratio,
        "rms_skewness": safe_skew(event_rms),
        "rms_kurtosis": safe_kurtosis(event_rms),
    }

    features.update(spectral_features(event_raw, fs))

    return features


def add_contextual_density_features(features_df):
    out = features_df.copy()

    if len(out) == 0:
        return out

    group_cols = ["recording_name", "mouse_id", "week", "segment_id"]
    group_cols = [c for c in group_cols if c in out.columns]

    out["segment_event_index"] = np.nan
    out["time_since_prev_event_sec"] = np.nan
    out["time_to_next_event_sec"] = np.nan
    out["local_event_count_pm60s"] = np.nan
    out["local_event_count_pm300s"] = np.nan
    out["event_density_per_segment_hour"] = np.nan
    out["segment_duration_sec_from_edf"] = np.nan

    # Cache EDF duration per path.
    edf_durations = {}
    for path in out["file_path_raw_signals"].dropna().unique():
        edf_durations[path] = get_edf_duration_sec(path)

    for _, idx in out.groupby(group_cols).groups.items():
        idx = list(idx)
        sub = out.loc[idx].copy().sort_values("start_sec")
        starts = sub["start_sec"].to_numpy(dtype=float)

        raw_path = str(sub["file_path_raw_signals"].iloc[0])
        segment_duration = edf_durations.get(raw_path, np.nan)

        if not np.isfinite(segment_duration) or segment_duration <= 0:
            segment_duration = float(np.nanmax(sub["end_sec"]) - np.nanmin(sub["start_sec"]))

        n_events = len(sub)
        density_per_h = n_events / (segment_duration / 3600) if segment_duration > 0 else np.nan

        ordered_idx = sub.index.to_numpy()

        for rank, original_idx in enumerate(ordered_idx):
            s = float(out.loc[original_idx, "start_sec"])

            prev_dt = s - starts[rank - 1] if rank > 0 else np.nan
            next_dt = starts[rank + 1] - s if rank < n_events - 1 else np.nan

            local60 = int(np.sum(np.abs(starts - s) <= 60)) - 1
            local300 = int(np.sum(np.abs(starts - s) <= 300)) - 1

            out.loc[original_idx, "segment_event_index"] = rank
            out.loc[original_idx, "time_since_prev_event_sec"] = prev_dt
            out.loc[original_idx, "time_to_next_event_sec"] = next_dt
            out.loc[original_idx, "local_event_count_pm60s"] = local60
            out.loc[original_idx, "local_event_count_pm300s"] = local300
            out.loc[original_idx, "event_density_per_segment_hour"] = density_per_h
            out.loc[original_idx, "segment_duration_sec_from_edf"] = segment_duration

    return out


def add_live_qc_if_available(events):
    out = events.copy()

    if VIDEO_QC_OUT.exists():
        qc = pd.read_csv(VIDEO_QC_OUT)

        if "stable_event_key" in out.columns and "stable_event_key" in qc.columns:
            keep = [c for c in ["stable_event_key", "qc_status", "qc_notes"] if c in qc.columns]
            out = out.merge(
                qc[keep].drop_duplicates("stable_event_key", keep="last"),
                on="stable_event_key",
                how="left",
            )
        elif "qc_event_id" in out.columns and "qc_event_id" in qc.columns:
            keep = [c for c in ["qc_event_id", "qc_status", "qc_notes"] if c in qc.columns]
            out = out.merge(
                qc[keep].drop_duplicates("qc_event_id", keep="last"),
                on="qc_event_id",
                how="left",
            )

    if "qc_status" not in out.columns:
        out["qc_status"] = "not_reviewed"
    else:
        out["qc_status"] = out["qc_status"].fillna("not_reviewed")

    if "qc_notes" not in out.columns:
        out["qc_notes"] = ""
    else:
        out["qc_notes"] = out["qc_notes"].fillna("")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pre-s", type=float, default=20.0)
    parser.add_argument("--post-s", type=float, default=20.0)
    parser.add_argument("--active-z", type=float, default=3.0)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    events_path = Path(args.events)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path)
    events = add_live_qc_if_available(events)

    if args.max_events is not None:
        events = events.head(args.max_events).copy()

    required = ["file_path_raw_signals", "start_sec", "end_sec"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"Missing required columns in event table: {missing}")

    rows = []
    failures = []

    print(f"Extracting EMG morphology features for {len(events)} events...")

    for i, (_, row) in enumerate(events.iterrows(), start=1):
        event_id = row.get("qc_event_id", i)

        try:
            feats = extract_one_event_features(
                row,
                pre_s=args.pre_s,
                post_s=args.post_s,
                active_z_threshold=args.active_z,
            )
            feats["feature_extraction_ok"] = True
            feats["feature_extraction_error"] = ""
            rows.append(feats)

        except Exception as e:
            fail = {
                "qc_event_id": event_id,
                "feature_extraction_ok": False,
                "feature_extraction_error": repr(e),
            }

            for c in [
                "recording_name",
                "group",
                "week",
                "mouse_id",
                "segment_id",
                "primary_category",
                "start_sec",
                "end_sec",
                "file_path_raw_signals",
            ]:
                fail[c] = row.get(c, "")

            rows.append(fail)
            failures.append(fail)

        if i % 25 == 0 or i == len(events):
            print(f"  {i}/{len(events)} done")

    features = pd.DataFrame(rows)
    features = add_contextual_density_features(features)

    out_path = out_dir / "emg_morphology_features.csv"
    features.to_csv(out_path, index=False)

    fail_path = out_dir / "emg_morphology_feature_extraction_failures.csv"
    pd.DataFrame(failures).to_csv(fail_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(fail_path)
    print(f"\nSuccessful events: {int(features['feature_extraction_ok'].sum()) if 'feature_extraction_ok' in features.columns else len(features)}")
    print(f"Failed events: {len(failures)}")


if __name__ == "__main__":
    main()
