from pathlib import Path
import argparse
import numpy as np
import pandas as pd

try:
    from pyedflib import EdfReader
except Exception:
    EdfReader = None


BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")

EEG_APP = Path.home() / "Desktop/local_sleep_manifests/final_WT_reference_manifests/WT_PD_week2_week21_EEGonly_finalWTref_application.csv"
FULL_APP = Path.home() / "Desktop/local_sleep_manifests/final_WT_reference_manifests/WT_PD_week2_week21_FULL_finalWTref_application.csv"

OUT_DIR = BASE / "EMG_burst_detection_NREM_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5

STATE_ORDER = ["Awake", "NREM", "REM"]


def normalize_state(x):
    x = str(x).strip()
    mapping = {
        "Wake": "Awake", "WK": "Awake", "W": "Awake", "AWAKE": "Awake", "wake": "Awake", "Awake": "Awake",
        "SWS": "NREM", "NREM": "NREM", "Nrem": "NREM",
        "PS": "REM", "REM": "REM", "Rem": "REM",
        "TR": "Undefined", "ND": "Undefined", "Undefined": "Undefined",
        "nan": "Undefined", "": "Undefined",
    }
    return mapping.get(x, x)


def load_stage_duration(path, n_epochs=None):
    path = Path(path)
    if not path.exists():
        return None

    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    states = []

    if not lines:
        return None

    if lines[0].startswith("*Duration"):
        prev_end = 0.0
        for line in lines[2:]:
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue

            label = " ".join(parts[:-1])
            end_sec = float(parts[-1])

            start_epoch = int(round(prev_end / EPOCH_SEC))
            end_epoch = int(round(end_sec / EPOCH_SEC))

            states.extend([normalize_state(label)] * max(0, end_epoch - start_epoch))
            prev_end = end_sec
    else:
        states = [normalize_state(x) for x in lines]

    if n_epochs is not None:
        if len(states) < n_epochs:
            states.extend(["Undefined"] * (n_epochs - len(states)))
        elif len(states) > n_epochs:
            states = states[:n_epochs]

    return np.array(states)


def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[s], dtype=float) for s in state_names]).T
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)
    return probs, state_names, pred, conf


def infer_eeg_emg_indices(labels):
    eeg_idx = None
    emg_idx = None

    for i, lab in enumerate(labels):
        u = str(lab).upper()
        if eeg_idx is None and "EEG" in u:
            eeg_idx = i
        if emg_idx is None and "EMG" in u:
            emg_idx = i

    if eeg_idx is None or emg_idx is None:
        raise ValueError(f"Could not infer EEG/EMG channels from EDF labels: {labels}")

    return eeg_idx, emg_idx


def read_emg_from_edf(edf_path):
    if EdfReader is None:
        raise ImportError("pyedflib is missing. Install with: pip install pyedflib")

    edf_path = Path(edf_path)

    with EdfReader(str(edf_path)) as reader:
        labels = reader.getSignalLabels()
        _, emg_idx = infer_eeg_emg_indices(labels)
        fs = float(reader.getSampleFrequency(emg_idx))
        n = int(reader.getNSamples()[emg_idx])
        emg = reader.readSignal(emg_idx, 0, n)

    return emg.astype(float), fs, labels[emg_idx]


def moving_rms(x, fs, window_sec=0.25):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)

    win = max(1, int(round(window_sec * fs)))
    kernel = np.ones(win) / win

    return np.sqrt(np.convolve(x ** 2, kernel, mode="same"))


def robust_scale_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826

    if not np.isfinite(mad) or mad <= 0:
        q25, q75 = np.percentile(x, [25, 75])
        mad = (q75 - q25) / 1.349

    if not np.isfinite(mad) or mad <= 0:
        mad = np.std(x)

    if not np.isfinite(mad) or mad <= 0:
        mad = 1.0

    return med, mad


def sample_mask_from_epoch_mask(epoch_mask, n_samples, fs):
    n_epochs = len(epoch_mask)
    sample_epochs = np.floor(np.arange(n_samples) / (fs * EPOCH_SEC)).astype(int)
    valid = sample_epochs < n_epochs
    mask = np.zeros(n_samples, dtype=bool)
    mask[valid] = epoch_mask[sample_epochs[valid]]
    return mask


def get_baseline_mask(eeg_probs, eeg_states, eeg_pred, n_samples, fs):
    n_epochs = len(eeg_pred)

    def state_prob(state):
        if state in eeg_states:
            return eeg_probs[:, eeg_states.index(state)]
        return np.zeros(n_epochs)

    p_nrem = state_prob("NREM")
    p_rem = state_prob("REM")
    p_wake = state_prob("Awake")

    highconf_nrem_epochs = (
        (p_nrem >= 0.80)
        & (p_rem <= 0.15)
        & (p_wake <= 0.15)
    )

    source = "high_conf_EEGonly_NREM"

    if highconf_nrem_epochs.sum() < 60:
        highconf_nrem_epochs = eeg_pred == "NREM"
        source = "EEGonly_predicted_NREM_fallback"

    sample_mask = sample_mask_from_epoch_mask(highconf_nrem_epochs, n_samples, fs)

    return sample_mask, source, int(highconf_nrem_epochs.sum())


def find_intervals_hysteresis(z, fs, onset_z=4.0, offset_z=2.0):
    above_on = z >= onset_z
    above_off = z >= offset_z

    intervals = []
    in_event = False
    start = None

    for i in range(len(z)):
        if not in_event and above_on[i]:
            start = i
            in_event = True
        elif in_event and not above_off[i]:
            end = i
            intervals.append((start, end))
            in_event = False
            start = None

    if in_event and start is not None:
        intervals.append((start, len(z) - 1))

    return intervals


def merge_intervals(intervals, fs, max_gap_sec):
    if not intervals:
        return []

    intervals = sorted(intervals)
    max_gap_samples = int(round(max_gap_sec * fs))

    merged = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        if start - cur_end <= max_gap_samples:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))
    return merged


def filter_min_duration(intervals, fs, min_sec):
    out = []
    for start, end in intervals:
        dur = (end - start + 1) / fs
        if dur >= min_sec:
            out.append((start, end))
    return out


def score_interval(row, start_s, end_s, eeg_probs, eeg_states, eeg_pred, full_probs, full_states, full_pred, manual):
    n_epochs = min(len(eeg_pred), len(full_pred))

    e0 = max(0, int(np.floor(start_s / EPOCH_SEC)))
    e1 = min(n_epochs, int(np.ceil(end_s / EPOCH_SEC)))

    if e1 <= e0:
        e1 = min(n_epochs, e0 + 1)

    center_epoch = min(n_epochs - 1, max(0, int(round(((start_s + end_s) / 2) / EPOCH_SEC))))

    def mean_prob(probs, states, state):
        if state not in states:
            return np.nan
        return float(np.mean(probs[e0:e1, states.index(state)]))

    p_rem_eeg = mean_prob(eeg_probs, eeg_states, "REM")
    p_wake_eeg = mean_prob(eeg_probs, eeg_states, "Awake")
    p_nrem_eeg = mean_prob(eeg_probs, eeg_states, "NREM")

    p_rem_full = mean_prob(full_probs, full_states, "REM")
    p_wake_full = mean_prob(full_probs, full_states, "Awake")
    p_nrem_full = mean_prob(full_probs, full_states, "NREM")

    # Distance to nearest EEG-only transition
    trans = np.where(eeg_pred[1:] != eeg_pred[:-1])[0] + 1
    if len(trans):
        dist_sec = float(np.min(np.abs(trans - center_epoch)) * EPOCH_SEC)
    else:
        dist_sec = np.inf

    if manual is not None and center_epoch < len(manual):
        manual_center = manual[center_epoch]
    else:
        manual_center = "Undefined"

    eeg_center = eeg_pred[center_epoch] if center_epoch < len(eeg_pred) else "Undefined"
    full_center = full_pred[center_epoch] if center_epoch < len(full_pred) else "Undefined"

    delta_rem = p_rem_eeg - p_rem_full

    # Simple category
    if (p_rem_eeg >= 0.70) and (delta_rem >= 0.25) and (p_rem_full <= 0.60):
        category = "EMG_suppressed_REM"
    elif (p_rem_eeg >= 0.70) and (p_wake_eeg <= 0.30) and (dist_sec >= 30):
        category = "stable_REM_EMG_burst"
    elif (
        ((0.30 <= p_rem_eeg < 0.70) and (0.20 <= p_wake_eeg <= 0.70))
        or (dist_sec < 30 and p_rem_eeg >= 0.30)
    ):
        category = "mixed_REM_Wake_transition"
    elif (p_wake_eeg >= 0.70) and (p_rem_eeg <= 0.30):
        category = "wake_like_movement"
    elif (p_nrem_eeg >= 0.70) and (p_rem_eeg < 0.30) and (p_wake_eeg < 0.70):
        category = "NREM_like_EMG"
    else:
        category = "other_uncertain"

    return {
        "start_sec": start_s,
        "end_sec": end_s,
        "duration_sec": end_s - start_s,
        "start_epoch": e0,
        "end_epoch": e1 - 1,
        "center_epoch": center_epoch,
        "manual_state_center": manual_center,
        "EEGonly_state_center": eeg_center,
        "full_state_center": full_center,
        "mean_EEGonly_P_REM": p_rem_eeg,
        "mean_EEGonly_P_Awake": p_wake_eeg,
        "mean_EEGonly_P_NREM": p_nrem_eeg,
        "mean_full_P_REM": p_rem_full,
        "mean_full_P_Awake": p_wake_full,
        "mean_full_P_NREM": p_nrem_full,
        "mean_delta_REM_EEGonly_minus_full": delta_rem,
        "min_EEGonly_distance_to_transition_sec": dist_sec,
        "event_class": category,
    }


def process_segment(row, full_row, args):
    raw_path = Path(row["file_path_raw_signals"])

    eeg_probs, eeg_states, eeg_pred, eeg_conf = load_probabilities(row["file_path_state_probabilities"])
    full_probs, full_states, full_pred, full_conf = load_probabilities(full_row["file_path_state_probabilities"])

    n_epochs = min(len(eeg_pred), len(full_pred))

    manual = None
    if "file_path_manual_state_annotation" in row and isinstance(row["file_path_manual_state_annotation"], str):
        manual = load_stage_duration(row["file_path_manual_state_annotation"], n_epochs=n_epochs)

    emg, fs, emg_label = read_emg_from_edf(raw_path)

    rms = moving_rms(emg, fs, window_sec=args.rms_window_sec)
    log_rms = np.log(rms + np.finfo(float).eps)

    baseline_mask, baseline_source, n_baseline_epochs = get_baseline_mask(
        eeg_probs,
        eeg_states,
        eeg_pred,
        n_samples=len(log_rms),
        fs=fs,
    )

    baseline_values = log_rms[baseline_mask]

    if len(baseline_values) < int(args.min_baseline_minutes * 60 * fs):
        # fallback: quietest part of whole recording
        baseline_source = "quietest_whole_recording_fallback"
        cutoff = np.percentile(log_rms, 30)
        baseline_values = log_rms[log_rms <= cutoff]

    # remove high EMG tail inside baseline to estimate low-tone baseline
    quiet_cutoff = np.percentile(baseline_values, args.baseline_quiet_percentile)
    quiet_baseline = baseline_values[baseline_values <= quiet_cutoff]

    baseline_median, baseline_scale = robust_scale_mad(quiet_baseline)
    z = (log_rms - baseline_median) / baseline_scale

    raw_intervals = find_intervals_hysteresis(
        z,
        fs,
        onset_z=args.onset_z,
        offset_z=args.offset_z,
    )

    micro = merge_intervals(raw_intervals, fs, max_gap_sec=args.micro_merge_gap_sec)
    micro = filter_min_duration(micro, fs, min_sec=args.min_micro_sec)

    episodes = merge_intervals(micro, fs, max_gap_sec=args.episode_gap_sec)
    episodes = filter_min_duration(episodes, fs, min_sec=args.min_episode_sec)

    micro_rows = []
    episode_rows = []

    for micro_id, (samp0, samp1) in enumerate(micro):
        start_s = samp0 / fs
        end_s = (samp1 + 1) / fs
        scored = score_interval(row, start_s, end_s, eeg_probs, eeg_states, eeg_pred, full_probs, full_states, full_pred, manual)

        segment_z = z[samp0:samp1 + 1]

        scored.update({
            "event_type": "microburst",
            "local_event_id": micro_id,
            "n_microbursts_in_episode": 1,
            "recording_name": row["recording_name"],
            "group": row["group"],
            "week": int(row["week"]),
            "mouse_id": int(row["mouse_id"]),
            "segment_id": int(row["segment_id"]),
            "emg_label": emg_label,
            "fs": fs,
            "baseline_source": baseline_source,
            "n_baseline_epochs": n_baseline_epochs,
            "baseline_median_log_rms": baseline_median,
            "baseline_scale_log_rms": baseline_scale,
            "max_EMG_baseline_z": float(np.max(segment_z)),
            "mean_EMG_baseline_z": float(np.mean(segment_z)),
            "file_path_raw_signals": row["file_path_raw_signals"],
            "file_path_state_probabilities_EEGonly": row["file_path_state_probabilities"],
            "file_path_state_probabilities_FULL": full_row["file_path_state_probabilities"],
            "file_path_manual_state_annotation": row.get("file_path_manual_state_annotation", ""),
        })
        micro_rows.append(scored)

    for episode_id, (samp0, samp1) in enumerate(episodes):
        start_s = samp0 / fs
        end_s = (samp1 + 1) / fs

        contained = [
            (m0, m1) for (m0, m1) in micro
            if m0 >= samp0 and m1 <= samp1
        ]

        total_micro_duration = sum((m1 - m0 + 1) / fs for m0, m1 in contained)
        episode_duration = end_s - start_s

        scored = score_interval(row, start_s, end_s, eeg_probs, eeg_states, eeg_pred, full_probs, full_states, full_pred, manual)

        segment_z = z[samp0:samp1 + 1]

        scored.update({
            "event_type": "episode",
            "local_event_id": episode_id,
            "n_microbursts_in_episode": len(contained),
            "total_microburst_duration_sec": total_micro_duration,
            "episode_duration_sec": episode_duration,
            "microburst_fraction_in_episode": total_micro_duration / episode_duration if episode_duration > 0 else np.nan,
            "microburst_rate_per_min_in_episode": len(contained) / (episode_duration / 60) if episode_duration > 0 else np.nan,
            "recording_name": row["recording_name"],
            "group": row["group"],
            "week": int(row["week"]),
            "mouse_id": int(row["mouse_id"]),
            "segment_id": int(row["segment_id"]),
            "emg_label": emg_label,
            "fs": fs,
            "baseline_source": baseline_source,
            "n_baseline_epochs": n_baseline_epochs,
            "baseline_median_log_rms": baseline_median,
            "baseline_scale_log_rms": baseline_scale,
            "max_EMG_baseline_z": float(np.max(segment_z)),
            "mean_EMG_baseline_z": float(np.mean(segment_z)),
            "file_path_raw_signals": row["file_path_raw_signals"],
            "file_path_state_probabilities_EEGonly": row["file_path_state_probabilities"],
            "file_path_state_probabilities_FULL": full_row["file_path_state_probabilities"],
            "file_path_manual_state_annotation": row.get("file_path_manual_state_annotation", ""),
        })
        episode_rows.append(scored)

    seg_summary = {
        "recording_name": row["recording_name"],
        "group": row["group"],
        "week": int(row["week"]),
        "mouse_id": int(row["mouse_id"]),
        "segment_id": int(row["segment_id"]),
        "duration_hours": len(emg) / fs / 3600,
        "baseline_source": baseline_source,
        "n_baseline_epochs": n_baseline_epochs,
        "baseline_median_log_rms": baseline_median,
        "baseline_scale_log_rms": baseline_scale,
        "n_microbursts": len(micro_rows),
        "n_episodes": len(episode_rows),
        "microbursts_per_hour": len(micro_rows) / (len(emg) / fs / 3600),
        "episodes_per_hour": len(episode_rows) / (len(emg) / fs / 3600),
    }

    return micro_rows, episode_rows, seg_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset-z", type=float, default=4.0)
    parser.add_argument("--offset-z", type=float, default=2.0)
    parser.add_argument("--rms-window-sec", type=float, default=0.25)
    parser.add_argument("--micro-merge-gap-sec", type=float, default=0.50)
    parser.add_argument("--episode-gap-sec", type=float, default=10.0)
    parser.add_argument("--min-micro-sec", type=float, default=0.10)
    parser.add_argument("--min-episode-sec", type=float, default=0.10)
    parser.add_argument("--baseline-quiet-percentile", type=float, default=80.0)
    parser.add_argument("--min-baseline-minutes", type=float, default=5.0)
    parser.add_argument("--only-mouse", type=int, default=None)
    parser.add_argument("--only-group", type=str, default="")
    parser.add_argument("--only-week", type=int, default=None)
    parser.add_argument("--eeg-manifest", type=Path, default=EEG_APP,
                        help="EEG-only Somnotate manifest CSV.")
    parser.add_argument("--full-manifest", type=Path, default=FULL_APP,
                        help="FULL EEG+EMG Somnotate manifest CSV.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory for EMG detection results.")
    args = parser.parse_args()

    globals()["OUT_DIR"] = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eeg = pd.read_csv(args.eeg_manifest)
    full = pd.read_csv(args.full_manifest)

    keys = ["recording_name", "group", "week", "mouse_id", "segment_id"]

    df = eeg.merge(
        full[keys + ["file_path_state_probabilities"]],
        on=keys,
        how="inner",
        suffixes=("", "_FULL"),
    )

    df = df.rename(columns={"file_path_state_probabilities_FULL": "full_prob_path"})

    if args.only_mouse is not None:
        df = df[df["mouse_id"].astype(int) == args.only_mouse].copy()
    if args.only_group:
        df = df[df["group"].astype(str) == args.only_group].copy()
    if args.only_week is not None:
        df = df[df["week"].astype(int) == args.only_week].copy()

    print("Segments to process:", len(df))
    print(pd.crosstab(df["group"], df["week"]))

    all_micro = []
    all_episodes = []
    summaries = []

    full_lookup = full.set_index(keys)

    for i, row in df.iterrows():
        key = tuple(row[k] for k in keys)
        full_row = full_lookup.loc[key].copy()
        full_row["file_path_state_probabilities"] = full_row["file_path_state_probabilities"]

        print(
            f"\nProcessing {i+1}/{len(df)}: "
            f"{row['recording_name']} | {row['group']} W{row['week']} | "
            f"mouse {row['mouse_id']} segment {row['segment_id']}"
        )

        try:
            micro_rows, episode_rows, summary = process_segment(row, full_row, args)
            all_micro.extend(micro_rows)
            all_episodes.extend(episode_rows)
            summaries.append(summary)

            print(
                f"  microbursts={summary['n_microbursts']} | "
                f"episodes={summary['n_episodes']} | "
                f"baseline={summary['baseline_source']}"
            )
        except Exception as e:
            print("  FAILED:", repr(e))
            summaries.append({
                "recording_name": row.get("recording_name", ""),
                "group": row.get("group", ""),
                "week": row.get("week", np.nan),
                "mouse_id": row.get("mouse_id", np.nan),
                "segment_id": row.get("segment_id", np.nan),
                "failed": True,
                "error": repr(e),
            })

    tag = f"NREMbaseline_on{args.onset_z:g}_off{args.offset_z:g}_episodegap{args.episode_gap_sec:g}s"

    micro_out = OUT_DIR / f"EMG_microbursts_{tag}.csv"
    episode_out = OUT_DIR / f"EMG_episodes_{tag}.csv"
    summary_out = OUT_DIR / f"EMG_baseline_detector_segment_summary_{tag}.csv"

    pd.DataFrame(all_micro).to_csv(micro_out, index=False)
    pd.DataFrame(all_episodes).to_csv(episode_out, index=False)
    pd.DataFrame(summaries).to_csv(summary_out, index=False)

    print("\nDone.")
    print("Wrote:")
    print(micro_out)
    print(episode_out)
    print(summary_out)

    if len(all_episodes):
        ep = pd.DataFrame(all_episodes)
        print("\nEpisode counts by group/week:")
        print(pd.crosstab(ep["group"], ep["week"]).to_string())
        print("\nEpisode classes:")
        print(ep["event_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
