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

OUT_DIR = Path.home() / "Desktop" / "PD21_interesting_REM_moments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5
WINDOW_MIN = 8
HALF_WINDOW_SEC = WINDOW_MIN * 60 / 2

REM_PROB_THRESHOLD = 0.8
STABLE_REM_MIN_SEC = 30
TRANSITION_EXCLUSION_SEC = 30
FMAX = 20

STATE_ORDER = ["Awake", "NREM", "REM", "Undefined"]
STATE_TO_CODE = {s: i for i, s in enumerate(STATE_ORDER)}
STATE_CMAP = ListedColormap(["#4e79a7", "#f28e2b", "#2ca25f", "#bdbdbd"])


# ---------------- HELPERS ----------------
def normalize_state(s):
    s = str(s).strip()
    mapping = {
        "Wake": "Awake", "W": "Awake", "AWAKE": "Awake", "wake": "Awake", "awake": "Awake",
        "NREM": "NREM", "Nrem": "NREM", "SWS": "NREM", "sws": "NREM", "NonREM": "NREM",
        "REM": "REM", "Rem": "REM", "PS": "REM", "ps": "REM", "Paradoxical Sleep": "REM",
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


def get_bouts(states, target_state="REM"):
    states = np.asarray(states)
    is_state = states == target_state

    starts = np.where(is_state & np.r_[True, ~is_state[:-1]])[0]
    ends = np.where(is_state & np.r_[~is_state[1:], True])[0]

    return list(zip(starts, ends))


def get_transition_epochs(states):
    states = np.asarray(states)
    return np.where(states[1:] != states[:-1])[0] + 1


def distance_to_nearest_transition(n_epochs, transition_epochs):
    if len(transition_epochs) == 0:
        return np.full(n_epochs, np.inf)

    idx = np.arange(n_epochs)
    dist = np.full(n_epochs, np.inf)

    for t in transition_epochs:
        dist = np.minimum(dist, np.abs(idx - t))

    return dist


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

        try:
            probs, state_names, pred, confidence, rem_prob = load_probabilities(prob_path)
        except Exception as e:
            print("Failed probability load:", row["recording_name"], row["segment_id"], repr(e))
            continue

        n_epochs = len(pred)
        rem_bouts = get_bouts(pred, "REM")
        transition_epochs = get_transition_epochs(pred)
        dist_to_transition_epochs = distance_to_nearest_transition(n_epochs, transition_epochs)

        for bout_id, (start, end) in enumerate(rem_bouts):
            dur_epochs = end - start + 1
            dur_sec = dur_epochs * EPOCH_SEC

            rp = rem_prob[start:end+1]
            conf = confidence[start:end+1]
            dtrans_sec = dist_to_transition_epochs[start:end+1] * EPOCH_SEC

            is_high_conf = np.mean(rp >= REM_PROB_THRESHOLD) >= 0.8
            is_far = np.min(dtrans_sec) >= TRANSITION_EXCLUSION_SEC
            is_long = dur_sec >= STABLE_REM_MIN_SEC

            start_sec = start * EPOCH_SEC
            end_sec = (end + 1) * EPOCH_SEC

            # Compute local EMG score for ranking interesting bouts.
            emg_p95, emg_mean = compute_local_emg_score(row, start_sec, end_sec)

            rows.append({
                "manifest_row": i,
                "recording_name": row["recording_name"],
                "mouse_id": row["mouse_id"],
                "week": row["week"],
                "segment_id": row["segment_id"],
                "bout_id": bout_id,
                "start_epoch": start,
                "end_epoch": end,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_min": start_sec / 60,
                "end_min": end_sec / 60,
                "duration_sec": dur_sec,
                "duration_min": dur_sec / 60,
                "mean_REM_prob": float(np.mean(rp)),
                "min_REM_prob": float(np.min(rp)),
                "max_REM_prob": float(np.max(rp)),
                "mean_confidence": float(np.mean(conf)),
                "min_confidence": float(np.min(conf)),
                "min_distance_to_transition_sec": float(np.min(dtrans_sec)),
                "stable_high_conf_REM": bool(is_high_conf and is_far and is_long),
                "short_REM_lt_30s": bool(dur_sec < 30),
                "local_emg_rms_p95": emg_p95,
                "local_emg_rms_mean": emg_mean,
                "file_path_state_probabilities": str(prob_path),
                "file_path_raw_signals": row["file_path_raw_signals"],
                "file_path_manual_state_annotation": row.get("file_path_manual_state_annotation", ""),
            })

    cand = pd.DataFrame(rows)

    if len(cand):
        # Bigger = more interesting EMG moment.
        cand["emg_rank"] = cand["local_emg_rms_p95"].rank(ascending=False, method="min")
        # Bigger = more ambiguous.
        cand["ambiguity_score"] = 1 - cand["mean_confidence"]
        cand["ambiguity_rank"] = cand["ambiguity_score"].rank(ascending=False, method="min")
        cand["duration_rank"] = cand["duration_sec"].rank(ascending=False, method="min")

    return cand


def select_examples(cand, max_each=2):
    examples = []

    def add_from(sub, category, n=max_each):
        nonlocal examples
        if len(sub) == 0:
            return

        for _, r in sub.head(n).iterrows():
            d = r.to_dict()
            d["example_category"] = category
            examples.append(d)

    if len(cand) == 0:
        return pd.DataFrame()

    # 1. Long stable REM
    stable = cand[cand["stable_high_conf_REM"]].sort_values(
        ["duration_sec", "mean_REM_prob"],
        ascending=[False, False]
    )
    add_from(stable, "long_stable_high_conf_REM")

    # 2. High EMG during REM
    high_emg = cand.sort_values("local_emg_rms_p95", ascending=False)
    add_from(high_emg, "high_EMG_during_REM")

    # 3. Ambiguous / mixed REM
    ambiguous = cand.sort_values(
        ["mean_confidence", "duration_sec"],
        ascending=[True, False]
    )
    add_from(ambiguous, "ambiguous_mixed_REM")

    # 4. Short REM
    short = cand[cand["short_REM_lt_30s"]].sort_values(
        ["mean_REM_prob", "duration_sec"],
        ascending=[False, True]
    )
    add_from(short, "short_REM_fragment")

    ex = pd.DataFrame(examples)

    if len(ex):
        ex = ex.drop_duplicates(
            subset=["recording_name", "segment_id", "bout_id", "start_sec", "example_category"]
        ).reset_index(drop=True)

    return ex


def plot_rem_example(row, example):
    prob_path = Path(row["file_path_state_probabilities"])
    raw_path = Path(row["file_path_raw_signals"])

    probs, state_names, pred, confidence, rem_prob = load_probabilities(prob_path)
    n_epochs = len(pred)

    manual = None
    manual_path = choose_manual_annotation_path(row.get("file_path_manual_state_annotation", ""))
    if manual_path is not None:
        try:
            manual = load_stage_duration(manual_path, n_epochs)
        except Exception:
            manual = None

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
        ax.axvline(rel_bout_start, linestyle="--", linewidth=1)
        ax.axvline(rel_bout_end, linestyle="--", linewidth=1)
        ax.axvspan(rel_bout_start, rel_bout_end, alpha=0.12)

    title = (
        f"{example['example_category']} | PD week 21 | mouse {row['mouse_id']} | "
        f"segment {row['segment_id']} | REM bout {int(example['bout_id'])}\n"
        f"duration={example['duration_sec']:.0f}s, mean REM prob={example['mean_REM_prob']:.2f}, "
        f"mean conf={example['mean_confidence']:.2f}, EMG p95={example['local_emg_rms_p95']:.2f} | "
        f"{row['recording_name']}"
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    safe_cat = str(example["example_category"]).replace(" ", "_")
    out = OUT_DIR / (
        f"{safe_cat}_mouse{row['mouse_id']}_seg{row['segment_id']}"
        f"_bout{int(example['bout_id'])}_{example['start_min']:.1f}min.png"
    )

    plt.savefig(out, dpi=180)
    plt.close()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Optional comma-separated row numbers from selected_PD21_REM_examples.csv to plot manually.",
    )
    parser.add_argument(
        "--max-each",
        type=int,
        default=2,
        help="Number of examples per category when auto-selecting.",
    )
    args = parser.parse_args()

    manifest = next((p for p in CANDIDATE_MANIFESTS if p.exists()), None)

    if manifest is None:
        raise FileNotFoundError("Could not find a final/pilot PD21 manifest.")

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
    cand_out = OUT_DIR / "PD21_REM_candidate_bouts_for_inspection.csv"
    candidates.to_csv(cand_out, index=False)

    print("\nWrote candidate table:")
    print(cand_out)

    if len(candidates) == 0:
        raise SystemExit("No PD21 REM bouts found.")

    selected = select_examples(candidates, max_each=args.max_each)
    selected_out = OUT_DIR / "selected_PD21_REM_examples.csv"
    selected.to_csv(selected_out, index=False)

    print("Wrote selected examples:")
    print(selected_out)

    if args.candidate_ids.strip():
        ids = [int(x.strip()) for x in args.candidate_ids.split(",") if x.strip()]
        selected = selected.iloc[ids].copy()

    print("\nSelected examples:")
    print(selected[[
        "example_category",
        "mouse_id",
        "segment_id",
        "bout_id",
        "start_min",
        "duration_sec",
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
            print("No matching manifest row for:", ex.to_dict())
            continue

        try:
            out = plot_rem_example(match.iloc[0], ex)
            print("Wrote:", out)
        except Exception as e:
            print("FAILED plot:", ex.to_dict())
            print("Error:", repr(e))

    print("\nDone.")
    print("Open folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
