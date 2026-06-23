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

OUT_DIR = Path.home() / "Desktop" / "PD21_manualWake_modelREM_moments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5
WINDOW_MIN = 8
HALF_WINDOW_SEC = WINDOW_MIN * 60 / 2

REM_PROB_THRESHOLD = 0.8
MIN_BOUT_SEC = 5
MIN_MANUAL_WAKE_FRACTION = 0.3
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

    if "REM" not in state_names:
        raise ValueError(f"No REM in {path}; keys={state_names}")

    rem_prob = probs[:, state_names.index("REM")]
    return probs, state_names, pred, confidence, rem_prob


def get_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    ends = np.where(mask & np.r_[~mask[1:], True])[0]
    return list(zip(starts, ends))


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

        if manual_path is None:
            print("No manual annotation for:", row["recording_name"], row["segment_id"])
            continue

        try:
            probs, state_names, pred, confidence, rem_prob = load_probabilities(prob_path)
            n_epochs = len(pred)
            manual = load_stage_duration(manual_path, n_epochs)
        except Exception as e:
            print("Failed load:", row["recording_name"], row["segment_id"], repr(e))
            continue

        # Core condition: model REM while manual Awake.
        model_rem_manual_awake = (
            (pred == "REM")
            & (manual == "Awake")
        )

        # Candidate runs are continuous periods where this condition is true.
        runs = get_true_runs(model_rem_manual_awake)

        for run_id, (start, end) in enumerate(runs):
            dur_epochs = end - start + 1
            dur_sec = dur_epochs * EPOCH_SEC

            if dur_sec < MIN_BOUT_SEC:
                continue

            # Extend to the full predicted REM bout containing this run.
            full_start = start
            while full_start > 0 and pred[full_start - 1] == "REM":
                full_start -= 1

            full_end = end
            while full_end < n_epochs - 1 and pred[full_end + 1] == "REM":
                full_end += 1

            full_dur_sec = (full_end - full_start + 1) * EPOCH_SEC

            manual_in_full = manual[full_start:full_end + 1]
            pred_in_full = pred[full_start:full_end + 1]
            rem_prob_full = rem_prob[full_start:full_end + 1]
            conf_full = confidence[full_start:full_end + 1]

            manual_wake_fraction_full = np.mean(manual_in_full == "Awake")
            manual_rem_fraction_full = np.mean(manual_in_full == "REM")
            manual_nrem_fraction_full = np.mean(manual_in_full == "NREM")

            if manual_wake_fraction_full < MIN_MANUAL_WAKE_FRACTION:
                continue

            start_sec = full_start * EPOCH_SEC
            end_sec = (full_end + 1) * EPOCH_SEC

            emg_p95, emg_mean = compute_local_emg_score(row, start_sec, end_sec)

            rows.append({
                "manifest_row": i,
                "recording_name": row["recording_name"],
                "mouse_id": row["mouse_id"],
                "week": row["week"],
                "segment_id": row["segment_id"],

                "run_id": run_id,
                "modelREM_manualAwake_start_epoch": start,
                "modelREM_manualAwake_end_epoch": end,
                "modelREM_manualAwake_duration_sec": dur_sec,
                "modelREM_manualAwake_duration_min": dur_sec / 60,

                "full_predREM_start_epoch": full_start,
                "full_predREM_end_epoch": full_end,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_min": start_sec / 60,
                "end_min": end_sec / 60,
                "full_predREM_duration_sec": full_dur_sec,
                "full_predREM_duration_min": full_dur_sec / 60,

                "manual_wake_fraction_full_predREM": manual_wake_fraction_full,
                "manual_rem_fraction_full_predREM": manual_rem_fraction_full,
                "manual_nrem_fraction_full_predREM": manual_nrem_fraction_full,

                "mean_REM_prob": float(np.mean(rem_prob_full)),
                "min_REM_prob": float(np.min(rem_prob_full)),
                "max_REM_prob": float(np.max(rem_prob_full)),
                "mean_confidence": float(np.mean(conf_full)),
                "min_confidence": float(np.min(conf_full)),

                "local_emg_rms_p95": emg_p95,
                "local_emg_rms_mean": emg_mean,

                "file_path_state_probabilities": str(prob_path),
                "file_path_raw_signals": row["file_path_raw_signals"],
                "file_path_manual_state_annotation": str(manual_path),
            })

    cand = pd.DataFrame(rows)

    if len(cand):
        # Rank interesting examples:
        # high manual-wake fraction, high REM probability, high EMG, and longer duration.
        cand["interesting_score"] = (
            2.0 * cand["manual_wake_fraction_full_predREM"]
            + 1.5 * cand["mean_REM_prob"]
            + 0.5 * (cand["full_predREM_duration_sec"] / cand["full_predREM_duration_sec"].max())
        )

        if cand["local_emg_rms_p95"].notna().any():
            emg_norm = cand["local_emg_rms_p95"] / cand["local_emg_rms_p95"].max()
            cand["interesting_score"] += 1.0 * emg_norm.fillna(0)

        cand = cand.sort_values(
            ["interesting_score", "local_emg_rms_p95", "mean_REM_prob"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return cand


def select_examples(cand, max_examples=8):
    if len(cand) == 0:
        return pd.DataFrame()

    selected = []

    # 1. Highest overall score
    selected.extend(cand.head(max_examples).to_dict("records"))

    # 2. Ensure high-EMG examples are represented
    high_emg = cand.sort_values("local_emg_rms_p95", ascending=False).head(3)
    selected.extend(high_emg.to_dict("records"))

    # 3. Ensure high manual wake fraction represented
    high_wake = cand.sort_values("manual_wake_fraction_full_predREM", ascending=False).head(3)
    selected.extend(high_wake.to_dict("records"))

    out = pd.DataFrame(selected)

    if len(out):
        out = out.drop_duplicates(
            subset=["recording_name", "segment_id", "full_predREM_start_epoch", "full_predREM_end_epoch"]
        ).head(max_examples).reset_index(drop=True)

        out["example_category"] = "manual_Wake_model_REM"

    return out


def plot_example(row, example):
    prob_path = Path(row["file_path_state_probabilities"])
    raw_path = Path(row["file_path_raw_signals"])
    manual_path = choose_manual_annotation_path(row.get("file_path_manual_state_annotation", ""))

    probs, state_names, pred, confidence, rem_prob = load_probabilities(prob_path)
    n_epochs = len(pred)
    manual = load_stage_duration(manual_path, n_epochs)

    bout_start_s = float(example["start_sec"])
    bout_end_s = float(example["end_sec"])
    center_s = (bout_start_s + bout_end_s) / 2

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

    rel_bout_start = (bout_start_s - center_s) / 60
    rel_bout_end = (bout_end_s - center_s) / 60

    # The exact sub-run where manual Wake + model REM overlap
    overlap_start_s = float(example["modelREM_manualAwake_start_epoch"]) * EPOCH_SEC
    overlap_end_s = (float(example["modelREM_manualAwake_end_epoch"]) + 1) * EPOCH_SEC
    rel_overlap_start = (overlap_start_s - center_s) / 60
    rel_overlap_end = (overlap_end_s - center_s) / 60

    fig, axes = plt.subplots(
        6, 1,
        figsize=(14, 11),
        gridspec_kw={"height_ratios": [0.55, 0.55, 1.5, 1.1, 1.1, 2.3]},
    )

    axes[0].imshow(
        state_bar(manual[epoch_start:epoch_end]),
        aspect="auto",
        interpolation="nearest",
        cmap=STATE_CMAP,
        vmin=0,
        vmax=len(STATE_ORDER) - 1,
        extent=rel_extent,
    )
    axes[0].set_yticks([])
    axes[0].set_ylabel("Manual")

    axes[1].imshow(
        state_bar(pred[epoch_start:epoch_end]),
        aspect="auto",
        interpolation="nearest",
        cmap=STATE_CMAP,
        vmin=0,
        vmax=len(STATE_ORDER) - 1,
        extent=rel_extent,
    )
    axes[1].set_yticks([])
    axes[1].set_ylabel("Somnotate")
    axes[1].text(
        1.01, 0.5,
        "Awake\nNREM\nREM\nUndefined",
        transform=axes[1].transAxes,
        va="center",
        fontsize=8,
    )

    for j, state in enumerate(state_names):
        axes[2].plot(
            rel_t_epoch_min,
            probs[epoch_start:epoch_end, j],
            label=state,
            linewidth=1.4,
        )

    axes[2].plot(
        rel_t_epoch_min,
        confidence[epoch_start:epoch_end],
        label="max prob",
        linewidth=1.0,
        linestyle="--",
    )
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("Probability")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(rel_t_signal_min, eeg_z, linewidth=0.45)
    axes[3].set_ylabel(f"EEG\n{eeg_label}\nrobust z")
    axes[3].set_xlim(-WINDOW_MIN / 2, WINDOW_MIN / 2)

    axes[4].plot(rel_t_signal_min, emg_z, linewidth=0.35, alpha=0.65, label="EMG raw")
    axes[4].plot(rel_t_signal_min, emg_rms_z, linewidth=1.0, label="EMG RMS")
    axes[4].set_ylabel(f"EMG\n{emg_label}\nrobust z")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].set_xlim(-WINDOW_MIN / 2, WINDOW_MIN / 2)

    axes[5].imshow(
        spec_img,
        origin="lower",
        aspect="auto",
        extent=[rel_t_spec_min.min(), rel_t_spec_min.max(), f.min(), f.max()],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[5].set_ylim(0, FMAX)
    axes[5].set_ylabel("EEG\nHz")
    axes[5].set_xlabel("Minutes relative to predicted REM bout center")

    for ax in axes:
        # full predicted REM bout
        ax.axvline(rel_bout_start, linestyle="--", linewidth=1)
        ax.axvline(rel_bout_end, linestyle="--", linewidth=1)
        ax.axvspan(rel_bout_start, rel_bout_end, alpha=0.10)

        # exact overlap manual Wake + model REM
        ax.axvspan(rel_overlap_start, rel_overlap_end, alpha=0.22)

    title = (
        f"Manual Wake / Somnotate REM | PD week 21 | mouse {row['mouse_id']} | "
        f"segment {row['segment_id']}\n"
        f"manual Wake fraction in predicted REM={example['manual_wake_fraction_full_predREM']:.2f}, "
        f"mean REM prob={example['mean_REM_prob']:.2f}, "
        f"duration={example['full_predREM_duration_sec']:.0f}s, "
        f"EMG p95={example['local_emg_rms_p95']:.2f} | "
        f"{row['recording_name']}"
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = OUT_DIR / (
        f"manualWake_modelREM_mouse{row['mouse_id']}_seg{row['segment_id']}"
        f"_{example['start_min']:.1f}min.png"
    )

    plt.savefig(out, dpi=180)
    plt.close()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="Number of examples to plot.",
    )
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Optional comma-separated row numbers from selected_manualWake_modelREM_examples.csv to plot manually.",
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
        & pd21["file_path_manual_state_annotation"].map(lambda p: choose_manual_annotation_path(p) is not None)
    ].copy()

    print("Using manifest:")
    print(manifest)
    print()
    print("PD week 21 rows with manual + probabilities + raw EDF:", len(pd21))
    print(pd21[["recording_name", "mouse_id", "segment_id"]].reset_index(drop=True).to_string())

    candidates = make_candidate_table(pd21)
    cand_out = OUT_DIR / "PD21_manualWake_modelREM_candidates.csv"
    candidates.to_csv(cand_out, index=False)

    print("\nWrote candidate table:")
    print(cand_out)

    if len(candidates) == 0:
        raise SystemExit(
            "No manual-Wake / model-REM candidates found. "
            "Try lowering MIN_MANUAL_WAKE_FRACTION or MIN_BOUT_SEC in the script."
        )

    selected = select_examples(candidates, max_examples=args.max_examples)
    selected_out = OUT_DIR / "selected_manualWake_modelREM_examples.csv"
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
        "start_min",
        "full_predREM_duration_sec",
        "modelREM_manualAwake_duration_sec",
        "manual_wake_fraction_full_predREM",
        "mean_REM_prob",
        "mean_confidence",
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
