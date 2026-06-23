from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import signal
from scipy.ndimage import gaussian_filter
from pyedflib import EdfReader


# ---------------- SETTINGS ----------------
CANDIDATE_MANIFESTS = [
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD21_GUI/PD21_finalWT_for_GUI.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz_completed.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv"),
]

OUT_DIR = Path.home() / "Desktop" / "PD21_NREM_REM_WAKE_sequences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5

# Window around the REM bout center
WINDOW_MIN = 10
HALF_WINDOW_SEC = WINDOW_MIN * 60 / 2

# What counts as a useful NREM→REM→Wake sequence
MIN_PRE_NREM_SEC = 60
MIN_REM_SEC = 10
MIN_POST_WAKE_SEC = 30

# Probability thresholds for ranking, not strict exclusion
NREM_PROB_THRESHOLD = 0.8
REM_PROB_THRESHOLD = 0.8
WAKE_PROB_THRESHOLD = 0.8

FMAX = 20

STATE_ORDER = ["Awake", "NREM", "REM", "Undefined"]
STATE_TO_CODE = {s: i for i, s in enumerate(STATE_ORDER)}
STATE_CMAP = ListedColormap(["#4e79a7", "#f28e2b", "#2ca25f", "#bdbdbd"])


# ---------------- HELPERS ----------------
def normalize_state(s):
    s = str(s).strip()
    mapping = {
        "Wake": "Awake", "WK": "Awake", "W": "Awake",
        "AWAKE": "Awake", "wake": "Awake", "awake": "Awake",

        "NREM": "NREM", "Nrem": "NREM", "SWS": "NREM", "sws": "NREM",
        "NonREM": "NREM",

        "REM": "REM", "Rem": "REM", "PS": "REM", "ps": "REM",
        "Paradoxical Sleep": "REM",

        "Undefined": "Undefined", "undefined": "Undefined", "ND": "Undefined",
        "TR": "Undefined", "nan": "Undefined", "NaN": "Undefined",
    }
    return mapping.get(s, s)


def choose_manual_annotation_path(original_path):
    if not isinstance(original_path, str) or not original_path:
        return None

    original = Path(original_path)
    candidates = [
        original.with_name("somnotate_annotation_lenfixed_lomo_512hz.tsv"),
        original.with_name("somnotate_annotation_lenfixed_512hz.tsv"),
        original,
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def load_stage_duration(path, n_epochs):
    if path is None:
        return None

    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    states = []

    if not lines:
        return np.array(["Undefined"] * n_epochs)

    if lines[0].startswith("*Duration"):
        prev_end_sec = 0.0

        for line in lines[2:]:
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue

            label = " ".join(parts[:-1])
            end_sec = float(parts[-1])

            start_epoch = int(round(prev_end_sec / EPOCH_SEC))
            end_epoch = int(round(end_sec / EPOCH_SEC))

            states.extend([normalize_state(label)] * max(0, end_epoch - start_epoch))
            prev_end_sec = end_sec
    else:
        states = [normalize_state(x) for x in lines]

    if len(states) < n_epochs:
        states.extend(["Undefined"] * (n_epochs - len(states)))
    elif len(states) > n_epochs:
        states = states[:n_epochs]

    return np.array(states)


def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[state], dtype=float) for state in state_names]).T
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)

    for required in ["Awake", "NREM", "REM"]:
        if required not in state_names:
            raise ValueError(f"{required} missing in {path}; keys={state_names}")

    return probs, state_names, pred, confidence


def get_bouts(states):
    states = np.asarray(states)
    starts = np.where(np.r_[True, states[1:] != states[:-1]])[0]
    ends = np.r_[starts[1:] - 1, len(states) - 1]
    bout_states = states[starts]
    return starts, ends, bout_states


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


def read_edf_window(edf_path, start_s, end_s):
    edf_path = Path(edf_path)

    with EdfReader(str(edf_path)) as reader:
        labels = reader.getSignalLabels()
        eeg_idx, emg_idx = infer_eeg_emg_indices(labels)

        fs = float(reader.getSampleFrequency(eeg_idx))

        start_sample = max(0, int(round(start_s * fs)))
        end_sample = max(start_sample + 1, int(round(end_s * fs)))
        n_samples = end_sample - start_sample

        eeg = reader.readSignal(eeg_idx, start_sample, n_samples)
        emg = reader.readSignal(emg_idx, start_sample, n_samples)

        t = np.arange(len(eeg)) / fs + start_s

    return t, eeg, emg, fs, labels[eeg_idx], labels[emg_idx]


def robust_z(x):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    scale = np.nanpercentile(np.abs(x), 95)

    if scale == 0 or not np.isfinite(scale):
        scale = np.nanstd(x)

    if scale == 0 or not np.isfinite(scale):
        scale = 1.0

    return np.clip(x / scale, -4, 4)


def emg_rms(emg, fs, window_sec=0.25):
    emg = np.asarray(emg, dtype=float)
    emg = emg - np.median(emg)

    win = max(1, int(round(window_sec * fs)))
    kernel = np.ones(win) / win

    return np.sqrt(np.convolve(emg ** 2, kernel, mode="same"))


def compute_spectrogram(eeg, fs, fmax=20, window_sec=1.0, overlap_fraction=0.9):
    eeg = np.asarray(eeg, dtype=float)
    eeg = eeg - np.median(eeg)

    nperseg = max(8, int(round(window_sec * fs)))
    noverlap = int(round(nperseg * overlap_fraction))
    nfft = int(round(fs * 4))

    f, t, Sxx = signal.spectrogram(
        eeg,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
        mode="psd",
    )

    keep = f <= fmax
    f = f[keep]
    Sxx = Sxx[keep, :]

    eps = np.finfo(float).eps
    logp = 10 * np.log10(Sxx + eps)

    mu = np.mean(logp, axis=1, keepdims=True)
    sd = np.std(logp, axis=1, keepdims=True)
    sd[sd == 0] = 1.0

    img = (logp - mu) / sd
    img = gaussian_filter(img, sigma=(0.6, 1.2))

    vmin = np.percentile(img, 5)
    vmax = np.percentile(img, 95)

    return f, t, img, vmin, vmax


def state_bar(states):
    codes = np.array([STATE_TO_CODE.get(s, STATE_TO_CODE["Undefined"]) for s in states])
    return codes.reshape(1, -1)


def compute_local_emg_score(row, start_sec, end_sec):
    raw_path = Path(row["file_path_raw_signals"])
    pad = 30
    start_s = max(0, start_sec - pad)
    end_s = end_sec + pad

    try:
        _, _, emg, fs, _, _ = read_edf_window(raw_path, start_s, end_s)
        rms = emg_rms(emg, fs)
        return float(np.percentile(rms, 95)), float(np.mean(rms))
    except Exception:
        return np.nan, np.nan


def make_candidate_table(df):
    rows = []

    for i, row in df.iterrows():
        prob_path = Path(row["file_path_state_probabilities"])
        manual_path = choose_manual_annotation_path(row.get("file_path_manual_state_annotation", ""))

        try:
            probs, state_names, pred, confidence = load_probabilities(prob_path)
            n_epochs = len(pred)
            manual = load_stage_duration(manual_path, n_epochs) if manual_path is not None else None
        except Exception as e:
            print("Failed load:", row["recording_name"], row["segment_id"], repr(e))
            continue

        state_idx = {s: state_names.index(s) for s in state_names}
        awake_prob = probs[:, state_idx["Awake"]]
        nrem_prob = probs[:, state_idx["NREM"]]
        rem_prob = probs[:, state_idx["REM"]]

        starts, ends, bout_states = get_bouts(pred)

        for bout_i, (start, end, state) in enumerate(zip(starts, ends, bout_states)):
            if state != "REM":
                continue

            if bout_i == 0 or bout_i == len(bout_states) - 1:
                continue

            prev_state = bout_states[bout_i - 1]
            next_state = bout_states[bout_i + 1]

            if prev_state != "NREM" or next_state != "Awake":
                continue

            prev_start, prev_end = starts[bout_i - 1], ends[bout_i - 1]
            next_start, next_end = starts[bout_i + 1], ends[bout_i + 1]

            pre_nrem_sec = (prev_end - prev_start + 1) * EPOCH_SEC
            rem_sec = (end - start + 1) * EPOCH_SEC
            post_wake_sec = (next_end - next_start + 1) * EPOCH_SEC

            if pre_nrem_sec < MIN_PRE_NREM_SEC:
                continue

            if rem_sec < MIN_REM_SEC:
                continue

            if post_wake_sec < MIN_POST_WAKE_SEC:
                continue

            start_sec = start * EPOCH_SEC
            end_sec = (end + 1) * EPOCH_SEC

            # Probabilities in each phase
            pre_nrem_prob_mean = float(np.mean(nrem_prob[prev_start:prev_end + 1]))
            rem_prob_mean = float(np.mean(rem_prob[start:end + 1]))
            rem_prob_min = float(np.min(rem_prob[start:end + 1]))
            wake_prob_after_mean = float(np.mean(awake_prob[next_start:next_end + 1]))

            mean_conf_rem = float(np.mean(confidence[start:end + 1]))
            min_conf_rem = float(np.min(confidence[start:end + 1]))

            emg_p95, emg_mean = compute_local_emg_score(row, start_sec, end_sec)

            manual_prev = ""
            manual_rem = ""
            manual_next = ""
            manual_wake_fraction_in_rem = np.nan
            manual_rem_fraction_in_rem = np.nan
            manual_nrem_fraction_in_rem = np.nan

            if manual is not None:
                manual_prev = pd.Series(manual[prev_start:prev_end + 1]).mode().iloc[0]
                manual_rem = pd.Series(manual[start:end + 1]).mode().iloc[0]
                manual_next = pd.Series(manual[next_start:next_end + 1]).mode().iloc[0]

                manual_wake_fraction_in_rem = float(np.mean(manual[start:end + 1] == "Awake"))
                manual_rem_fraction_in_rem = float(np.mean(manual[start:end + 1] == "REM"))
                manual_nrem_fraction_in_rem = float(np.mean(manual[start:end + 1] == "NREM"))

            # Higher = more canonical NREM->REM->Wake, with strong EMG included.
            score = (
                pre_nrem_prob_mean
                + rem_prob_mean
                + wake_prob_after_mean
                + 0.25 * np.log1p(rem_sec)
            )

            rows.append({
                "manifest_row": i,
                "recording_name": row["recording_name"],
                "mouse_id": row["mouse_id"],
                "week": row["week"],
                "segment_id": row["segment_id"],

                "pred_sequence": "NREM->REM->Awake",
                "rem_bout_i": bout_i,

                "prev_nrem_start_epoch": int(prev_start),
                "prev_nrem_end_epoch": int(prev_end),
                "rem_start_epoch": int(start),
                "rem_end_epoch": int(end),
                "next_wake_start_epoch": int(next_start),
                "next_wake_end_epoch": int(next_end),

                "rem_start_sec": start_sec,
                "rem_end_sec": end_sec,
                "rem_start_min": start_sec / 60,
                "rem_end_min": end_sec / 60,

                "pre_nrem_sec": pre_nrem_sec,
                "rem_duration_sec": rem_sec,
                "post_wake_sec": post_wake_sec,

                "pre_nrem_prob_mean": pre_nrem_prob_mean,
                "rem_prob_mean": rem_prob_mean,
                "rem_prob_min": rem_prob_min,
                "wake_prob_after_mean": wake_prob_after_mean,
                "mean_confidence_REM": mean_conf_rem,
                "min_confidence_REM": min_conf_rem,

                "manual_prev_mode": manual_prev,
                "manual_during_model_REM_mode": manual_rem,
                "manual_next_mode": manual_next,
                "manual_wake_fraction_during_model_REM": manual_wake_fraction_in_rem,
                "manual_rem_fraction_during_model_REM": manual_rem_fraction_in_rem,
                "manual_nrem_fraction_during_model_REM": manual_nrem_fraction_in_rem,

                "local_emg_rms_p95": emg_p95,
                "local_emg_rms_mean": emg_mean,

                "candidate_score": score,

                "file_path_state_probabilities": str(prob_path),
                "file_path_raw_signals": row["file_path_raw_signals"],
                "file_path_manual_state_annotation": str(manual_path) if manual_path is not None else "",
            })

    cand = pd.DataFrame(rows)

    if len(cand):
        if cand["local_emg_rms_p95"].notna().any():
            emg_norm = cand["local_emg_rms_p95"] / cand["local_emg_rms_p95"].max()
            cand["candidate_score"] += emg_norm.fillna(0)

        cand = cand.sort_values(
            ["candidate_score", "rem_duration_sec", "local_emg_rms_p95"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return cand


def select_examples(cand, max_examples=8):
    if len(cand) == 0:
        return pd.DataFrame()

    selected = []

    # Best canonical NREM->REM->Wake sequences
    selected.extend(cand.head(max_examples).to_dict("records"))

    # Ensure high-EMG REM transitions included
    high_emg = cand.sort_values("local_emg_rms_p95", ascending=False).head(3)
    selected.extend(high_emg.to_dict("records"))

    # Ensure manual Wake during model REM included, if present
    if "manual_wake_fraction_during_model_REM" in cand.columns:
        high_manual_wake = cand.sort_values(
            "manual_wake_fraction_during_model_REM",
            ascending=False
        ).head(3)
        selected.extend(high_manual_wake.to_dict("records"))

    out = pd.DataFrame(selected)

    if len(out):
        out = out.drop_duplicates(
            subset=["recording_name", "segment_id", "rem_start_epoch", "rem_end_epoch"]
        ).head(max_examples).reset_index(drop=True)

        out["example_category"] = "NREM_REM_WAKE_sequence"

    return out


def plot_example(row, example):
    prob_path = Path(row["file_path_state_probabilities"])
    raw_path = Path(row["file_path_raw_signals"])
    manual_path = choose_manual_annotation_path(row.get("file_path_manual_state_annotation", ""))

    probs, state_names, pred, confidence = load_probabilities(prob_path)
    n_epochs = len(pred)
    manual = load_stage_duration(manual_path, n_epochs) if manual_path is not None else None

    rem_start_s = float(example["rem_start_sec"])
    rem_end_s = float(example["rem_end_sec"])
    center_s = (rem_start_s + rem_end_s) / 2

    start_s = max(0.0, center_s - HALF_WINDOW_SEC)
    end_s = center_s + HALF_WINDOW_SEC

    t_sig, eeg, emg, fs, eeg_label, emg_label = read_edf_window(raw_path, start_s, end_s)

    rel_t_signal_min = (t_sig - center_s) / 60

    eeg_z = robust_z(eeg)
    emg_z = robust_z(emg)
    emg_rms_z = robust_z(emg_rms(emg, fs))

    f, t_spec, spec_img, vmin, vmax = compute_spectrogram(eeg, fs, fmax=FMAX)
    rel_t_spec_min = (t_spec + start_s - center_s) / 60

    epoch_start = max(0, int(np.floor(start_s / EPOCH_SEC)))
    epoch_end = min(n_epochs, int(np.ceil(end_s / EPOCH_SEC)))

    rel_t_epoch_min = (np.arange(epoch_start, epoch_end) * EPOCH_SEC - center_s) / 60
    rel_extent = [
        rel_t_epoch_min[0],
        rel_t_epoch_min[-1] + EPOCH_SEC / 60,
        0,
        1,
    ]

    # Sequence spans
    rel_prev_start = (float(example["prev_nrem_start_epoch"]) * EPOCH_SEC - center_s) / 60
    rel_prev_end = ((float(example["prev_nrem_end_epoch"]) + 1) * EPOCH_SEC - center_s) / 60

    rel_rem_start = (rem_start_s - center_s) / 60
    rel_rem_end = (rem_end_s - center_s) / 60

    rel_wake_start = (float(example["next_wake_start_epoch"]) * EPOCH_SEC - center_s) / 60
    rel_wake_end = ((float(example["next_wake_end_epoch"]) + 1) * EPOCH_SEC - center_s) / 60

    has_manual = manual is not None

    if has_manual:
        fig, axes = plt.subplots(
            6, 1,
            figsize=(14, 11),
            gridspec_kw={"height_ratios": [0.55, 0.55, 1.5, 1.1, 1.1, 2.3]},
        )
    else:
        fig, axes = plt.subplots(
            5, 1,
            figsize=(14, 10),
            gridspec_kw={"height_ratios": [0.55, 1.5, 1.1, 1.1, 2.3]},
        )

    ax_i = 0

    if has_manual:
        axes[ax_i].imshow(
            state_bar(manual[epoch_start:epoch_end]),
            aspect="auto",
            interpolation="nearest",
            cmap=STATE_CMAP,
            vmin=0,
            vmax=len(STATE_ORDER) - 1,
            extent=rel_extent,
        )
        axes[ax_i].set_yticks([])
        axes[ax_i].set_ylabel("Manual")
        ax_i += 1

    axes[ax_i].imshow(
        state_bar(pred[epoch_start:epoch_end]),
        aspect="auto",
        interpolation="nearest",
        cmap=STATE_CMAP,
        vmin=0,
        vmax=len(STATE_ORDER) - 1,
        extent=rel_extent,
    )
    axes[ax_i].set_yticks([])
    axes[ax_i].set_ylabel("Somnotate")
    axes[ax_i].text(
        1.01, 0.5,
        "Awake\nNREM\nREM\nUndefined",
        transform=axes[ax_i].transAxes,
        va="center",
        fontsize=8,
    )
    ax_i += 1

    for j, state in enumerate(state_names):
        axes[ax_i].plot(
            rel_t_epoch_min,
            probs[epoch_start:epoch_end, j],
            label=state,
            linewidth=1.4,
        )

    axes[ax_i].plot(
        rel_t_epoch_min,
        confidence[epoch_start:epoch_end],
        label="max prob",
        linewidth=1.0,
        linestyle="--",
    )
    axes[ax_i].set_ylim(-0.02, 1.02)
    axes[ax_i].set_ylabel("Probability")
    axes[ax_i].legend(loc="upper right", fontsize=8)
    ax_i += 1

    axes[ax_i].plot(rel_t_signal_min, eeg_z, linewidth=0.45)
    axes[ax_i].set_ylabel(f"EEG\n{eeg_label}\nrobust z")
    axes[ax_i].set_xlim(-WINDOW_MIN / 2, WINDOW_MIN / 2)
    ax_i += 1

    axes[ax_i].plot(rel_t_signal_min, emg_z, linewidth=0.35, alpha=0.65, label="EMG raw")
    axes[ax_i].plot(rel_t_signal_min, emg_rms_z, linewidth=1.0, label="EMG RMS")
    axes[ax_i].set_ylabel(f"EMG\n{emg_label}\nrobust z")
    axes[ax_i].legend(loc="upper right", fontsize=8)
    axes[ax_i].set_xlim(-WINDOW_MIN / 2, WINDOW_MIN / 2)
    ax_i += 1

    axes[ax_i].imshow(
        spec_img,
        origin="lower",
        aspect="auto",
        extent=[rel_t_spec_min.min(), rel_t_spec_min.max(), f.min(), f.max()],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[ax_i].set_ylim(0, FMAX)
    axes[ax_i].set_ylabel("EEG\nHz")
    axes[ax_i].set_xlabel("Minutes relative to REM bout center")

    for ax in axes:
        # predicted NREM before
        ax.axvspan(rel_prev_start, rel_prev_end, alpha=0.08)

        # predicted REM
        ax.axvspan(rel_rem_start, rel_rem_end, alpha=0.18)

        # predicted wake after
        ax.axvspan(rel_wake_start, rel_wake_end, alpha=0.08)

        ax.axvline(rel_rem_start, linestyle="--", linewidth=1)
        ax.axvline(rel_rem_end, linestyle="--", linewidth=1)

    title = (
        f"Predicted NREM → REM → Wake | PD week 21 | mouse {row['mouse_id']} | "
        f"segment {row['segment_id']}\n"
        f"NREM before={example['pre_nrem_sec']:.0f}s, REM={example['rem_duration_sec']:.0f}s, "
        f"Wake after={example['post_wake_sec']:.0f}s, "
        f"REM prob={example['rem_prob_mean']:.2f}, "
        f"EMG p95={example['local_emg_rms_p95']:.2f} | "
        f"{row['recording_name']}"
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = OUT_DIR / (
        f"NREM_REM_WAKE_mouse{row['mouse_id']}_seg{row['segment_id']}"
        f"_{example['rem_start_min']:.1f}min.png"
    )

    plt.savefig(out, dpi=180)
    plt.close()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Optional comma-separated row numbers from selected_NREM_REM_WAKE_examples.csv.",
    )
    args = parser.parse_args()

    manifest = next((p for p in CANDIDATE_MANIFESTS if p.exists()), None)

    if manifest is None:
        raise FileNotFoundError("Could not find final/pilot manifest.")

    df = pd.read_csv(manifest)

    pd21 = df[
        (df["group"] == "PD")
        & (df["week"] == 21)
    ].copy()

    pd21 = pd21[
        pd21["file_path_state_probabilities"].map(lambda p: Path(p).exists())
        & pd21["file_path_raw_signals"].map(lambda p: Path(p).exists())
    ].copy()

    print("Using manifest:")
    print(manifest)
    print()
    print("PD week 21 rows:", len(pd21))
    print(pd21[["recording_name", "mouse_id", "segment_id"]].reset_index(drop=True).to_string())

    candidates = make_candidate_table(pd21)
    cand_out = OUT_DIR / "PD21_NREM_REM_WAKE_candidates.csv"
    candidates.to_csv(cand_out, index=False)

    print("\nWrote candidate table:")
    print(cand_out)

    if len(candidates) == 0:
        raise SystemExit(
            "No NREM→REM→Wake candidates found. "
            "Try lowering MIN_PRE_NREM_SEC, MIN_REM_SEC, or MIN_POST_WAKE_SEC in the script."
        )

    selected = select_examples(candidates, max_examples=args.max_examples)
    selected_out = OUT_DIR / "selected_NREM_REM_WAKE_examples.csv"
    selected.to_csv(selected_out, index=False)

    print("Wrote selected examples:")
    print(selected_out)

    if args.candidate_ids.strip():
        ids = [int(x.strip()) for x in args.candidate_ids.split(",") if x.strip()]
        selected = selected.iloc[ids].copy()

    print("\nSelected examples:")
    print(selected[[
        "mouse_id",
        "segment_id",
        "rem_start_min",
        "pre_nrem_sec",
        "rem_duration_sec",
        "post_wake_sec",
        "rem_prob_mean",
        "mean_confidence_REM",
        "manual_during_model_REM_mode",
        "manual_wake_fraction_during_model_REM",
        "local_emg_rms_p95",
    ]].to_string(index=True))

    print("\nCreating figures...")

    for _, ex in selected.iterrows():
        match = pd21[
            (pd21["recording_name"] == ex["recording_name"])
            & (pd21["mouse_id"] == ex["mouse_id"])
            & (pd21["segment_id"] == ex["segment_id"])
        ]

        if len(match) == 0:
            print("No matching row for:", ex.to_dict())
            continue

        try:
            out = plot_example(match.iloc[0], ex)
            print("Wrote:", out)
        except Exception as e:
            print("FAILED plot:", ex.to_dict())
            print("Error:", repr(e))

    print("\nDone.")
    print("Open folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
