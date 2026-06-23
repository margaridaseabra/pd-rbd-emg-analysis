from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import signal
from scipy.ndimage import gaussian_filter
from pyedflib import EdfReader


# ---------------- SETTINGS ----------------
MANIFEST = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv"
)

OUT_DIR = Path.home() / "Desktop" / "somnotate_transition_figures_with_eeg_emg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5
WINDOW_MIN = 8
HALF_WINDOW_SEC = WINDOW_MIN * 60 / 2

LOCAL_SCORE_SEC = 60
LOCAL_SCORE_EPOCHS = int(LOCAL_SCORE_SEC / EPOCH_SEC)

FMAX = 20

PRIORITY_TRANSITIONS = [
    "Awake->NREM",
    "NREM->REM",
    "REM->Awake",
    "NREM->Awake",
]

STATE_ORDER = ["Awake", "NREM", "REM", "Undefined"]
STATE_TO_CODE = {s: i for i, s in enumerate(STATE_ORDER)}
STATE_COLORS = ["#4e79a7", "#f28e2b", "#2ca25f", "#bdbdbd"]
STATE_CMAP = ListedColormap(STATE_COLORS)


# ---------------- HELPERS ----------------
def normalize_state(s):
    s = str(s).strip()

    mapping = {
        "Wake": "Awake",
        "W": "Awake",
        "AWAKE": "Awake",
        "wake": "Awake",
        "awake": "Awake",

        "NREM": "NREM",
        "Nrem": "NREM",
        "SWS": "NREM",
        "sws": "NREM",
        "NonREM": "NREM",

        "REM": "REM",
        "Rem": "REM",
        "PS": "REM",
        "ps": "REM",
        "Paradoxical Sleep": "REM",

        "Undefined": "Undefined",
        "undefined": "Undefined",
        "ND": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
    }

    return mapping.get(s, s)


def choose_manual_annotation_path(original_path):
    original = Path(original_path)

    candidates = [
        original.with_name("somnotate_annotation_lenfixed_lomo_512hz.tsv"),
        original.with_name("somnotate_annotation_lenfixed_512hz.tsv"),
        original,
    ]

    for p in candidates:
        if p.exists():
            return p

    return original


def load_stage_duration(path, n_epochs):
    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

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
    return probs, state_names, pred, confidence


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

        fs_eeg = float(reader.getSampleFrequency(eeg_idx))
        fs_emg = float(reader.getSampleFrequency(emg_idx))

        if abs(fs_eeg - fs_emg) > 1e-6:
            print(f"Warning: EEG fs={fs_eeg}, EMG fs={fs_emg}. Using EEG fs.")

        fs = fs_eeg

        start_sample = max(0, int(round(start_s * fs)))
        end_sample = int(round(end_s * fs))
        n_samples = max(1, end_sample - start_sample)

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
    return np.clip(x / scale, -3, 3)


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

    # Per-frequency z-score to make bands visible
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


def find_transition_candidates(row):
    prob_path = Path(row["file_path_state_probabilities"])
    probs, state_names, pred, confidence = load_probabilities(prob_path)

    n_epochs = len(pred)
    change_points = np.where(pred[1:] != pred[:-1])[0] + 1

    candidates = []

    manual = None
    if "file_path_manual_state_annotation" in row and isinstance(row["file_path_manual_state_annotation"], str):
        manual_path = choose_manual_annotation_path(row["file_path_manual_state_annotation"])
        if manual_path.exists():
            try:
                manual = load_stage_duration(manual_path, n_epochs)
            except Exception:
                manual = None

    for cp in change_points:
        before = pred[cp - 1]
        after = pred[cp]
        transition_type = f"{before}->{after}"

        lo = max(0, cp - LOCAL_SCORE_EPOCHS)
        hi = min(n_epochs, cp + LOCAL_SCORE_EPOCHS)

        local_probs = probs[lo:hi]
        local_conf = confidence[lo:hi]

        ambiguity_score = 1.0 - np.mean(np.max(local_probs, axis=1))
        min_conf = np.min(local_conf)
        mean_conf = np.mean(local_conf)

        local_manual_agreement = np.nan
        manual_before = ""
        manual_after = ""

        if manual is not None:
            manual_before = manual[max(0, cp - 1)]
            manual_after = manual[min(n_epochs - 1, cp)]

            valid = np.isin(manual[lo:hi], ["Awake", "NREM", "REM"])
            if np.any(valid):
                local_manual_agreement = np.mean(manual[lo:hi][valid] == pred[lo:hi][valid]) * 100

        candidates.append({
            "recording_name": row["recording_name"],
            "group": row.get("group", ""),
            "week": row.get("week", ""),
            "mouse_id": row["mouse_id"],
            "segment_id": row["segment_id"],
            "transition_epoch": int(cp),
            "transition_sec": cp * EPOCH_SEC,
            "transition_min_from_start": cp * EPOCH_SEC / 60,
            "transition_type": transition_type,
            "before_state": before,
            "after_state": after,
            "ambiguity_score": ambiguity_score,
            "min_conf_local": min_conf,
            "mean_conf_local": mean_conf,
            "local_manual_agreement_pct": local_manual_agreement,
            "manual_before": manual_before,
            "manual_after": manual_after,
        })

    return candidates


def make_transition_figure(row, cand):
    prob_path = Path(row["file_path_state_probabilities"])
    edf_path = Path(row["file_path_raw_signals"])

    probs, state_names, pred, confidence = load_probabilities(prob_path)
    n_epochs = len(pred)

    manual = None
    manual_path = choose_manual_annotation_path(row["file_path_manual_state_annotation"])
    if manual_path.exists():
        try:
            manual = load_stage_duration(manual_path, n_epochs)
        except Exception:
            manual = None

    center_s = float(cand["transition_sec"])
    start_s = max(0.0, center_s - HALF_WINDOW_SEC)
    end_s = center_s + HALF_WINDOW_SEC

    t_signal, eeg, emg, fs, eeg_label, emg_label = read_edf_window(edf_path, start_s, end_s)

    rel_t_signal_min = (t_signal - center_s) / 60

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

    has_manual = manual is not None

    if has_manual:
        fig, axes = plt.subplots(
            6,
            1,
            figsize=(14, 11),
            sharex=False,
            gridspec_kw={"height_ratios": [0.55, 0.55, 1.5, 1.1, 1.1, 2.3]},
        )
    else:
        fig, axes = plt.subplots(
            5,
            1,
            figsize=(14, 10),
            sharex=False,
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
        axes[ax_i].axvline(0, linestyle="--", linewidth=1)
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
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)

    # Add legend-like text
    axes[ax_i].text(
        1.01,
        0.5,
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
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)
    ax_i += 1

    axes[ax_i].plot(rel_t_signal_min, eeg_z, linewidth=0.45)
    axes[ax_i].set_ylabel(f"EEG\n{eeg_label}\nrobust z")
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)
    axes[ax_i].set_xlim(-WINDOW_MIN / 2, WINDOW_MIN / 2)
    ax_i += 1

    axes[ax_i].plot(rel_t_signal_min, emg_z, linewidth=0.35, alpha=0.65, label="EMG raw")
    axes[ax_i].plot(rel_t_signal_min, emg_rms_z, linewidth=1.0, label="EMG RMS")
    axes[ax_i].set_ylabel(f"EMG\n{emg_label}\nrobust z")
    axes[ax_i].legend(loc="upper right", fontsize=8)
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)
    axes[ax_i].set_xlim(-WINDOW_MIN / 2, WINDOW_MIN / 2)
    ax_i += 1

    axes[ax_i].imshow(
        spec_img,
        origin="lower",
        aspect="auto",
        extent=[
            rel_t_spec_min.min(),
            rel_t_spec_min.max(),
            f.min(),
            f.max(),
        ],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[ax_i].set_ylim(0, FMAX)
    axes[ax_i].set_ylabel("EEG\nHz")
    axes[ax_i].set_xlabel("Minutes relative to transition")
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)

    title = (
        f"{cand['transition_type']} transition | "
        f"mouse {row['mouse_id']} | week {row.get('week', '')} | "
        f"segment {row['segment_id']} | "
        f"t={cand['transition_min_from_start']:.1f} min\n"
        f"ambiguity={cand['ambiguity_score']:.3f}, "
        f"min confidence={cand['min_conf_local']:.3f}, "
        f"local manual agreement={cand['local_manual_agreement_pct']:.1f}% | "
        f"{row['recording_name']}"
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    safe_transition = cand["transition_type"].replace("->", "_to_")
    out = OUT_DIR / (
        f"{safe_transition}_mouse{row['mouse_id']}_week{row.get('week','')}"
        f"_seg{row['segment_id']}_{cand['transition_min_from_start']:.1f}min_with_signals.png"
    )

    plt.savefig(out, dpi=180)
    plt.close()

    return out


# ---------------- MAIN ----------------
df = pd.read_csv(MANIFEST)

# Prefer WT manually scored segments for presentation examples.
manual_exists = df["file_path_manual_state_annotation"].map(
    lambda p: Path(p).exists() if isinstance(p, str) else False
)
prob_exists = df["file_path_state_probabilities"].map(
    lambda p: Path(p).exists() if isinstance(p, str) else False
)
edf_exists = df["file_path_raw_signals"].map(
    lambda p: Path(p).exists() if isinstance(p, str) else False
)

pool = df[
    (df["group"] == "WT")
    & (df["week"].isin([2, 21]))
    & manual_exists
    & prob_exists
    & edf_exists
].copy()

if "pct_scored" in pool.columns:
    pool = pool[pool["pct_scored"] >= 0.90].copy()

print("Using WT manually scored rows:", len(pool))
print(pool[["recording_name", "mouse_id", "week", "segment_id"]].to_string(index=False))

all_candidates = []

for _, row in pool.iterrows():
    try:
        all_candidates.extend(find_transition_candidates(row))
    except Exception as e:
        print("FAILED candidate extraction:", row["recording_name"], row["segment_id"], repr(e))

cand_df = pd.DataFrame(all_candidates)

if len(cand_df) == 0:
    raise SystemExit("No transition candidates found.")

cand_df = cand_df.sort_values(
    ["ambiguity_score", "min_conf_local"],
    ascending=[False, True],
).reset_index(drop=True)

cand_out = OUT_DIR / "transition_candidates_with_signal_summary.csv"
cand_df.to_csv(cand_out, index=False)
print("\nWrote:", cand_out)

# Choose one strong example for each transition type.
chosen = []
used_exact = set()

for transition_type in PRIORITY_TRANSITIONS:
    sub = cand_df[cand_df["transition_type"] == transition_type].copy()
    if len(sub) == 0:
        print("No candidates for:", transition_type)
        continue

    c = sub.iloc[0]
    chosen.append(c)
    used_exact.add((c["recording_name"], c["segment_id"], c["transition_epoch"]))

# Add one most ambiguous overall if different.
for _, c in cand_df.iterrows():
    key = (c["recording_name"], c["segment_id"], c["transition_epoch"])
    if key not in used_exact:
        chosen.append(c)
        break

chosen_df = pd.DataFrame(chosen)
chosen_out = OUT_DIR / "chosen_transition_signal_examples.csv"
chosen_df.to_csv(chosen_out, index=False)
print("Wrote:", chosen_out)

print("\nCreating figures...")

for _, c in chosen_df.iterrows():
    match = pool[
        (pool["recording_name"] == c["recording_name"])
        & (pool["mouse_id"] == c["mouse_id"])
        & (pool["week"] == c["week"])
        & (pool["segment_id"] == c["segment_id"])
    ]

    if len(match) == 0:
        print("Could not find matching row for candidate:", c.to_dict())
        continue

    row = match.iloc[0]

    try:
        out = make_transition_figure(row, c)
        print("Wrote:", out)
    except Exception as e:
        print("FAILED figure:", c["transition_type"], c["recording_name"], c["segment_id"], repr(e))

print("\nTop transition candidates:")
print(
    cand_df[
        [
            "transition_type",
            "mouse_id",
            "week",
            "segment_id",
            "transition_min_from_start",
            "ambiguity_score",
            "min_conf_local",
            "local_manual_agreement_pct",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nDone.")
print("Open folder:")
print(OUT_DIR)
