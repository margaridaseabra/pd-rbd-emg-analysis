from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MANIFEST = Path("local_manifests/wt_week2_train_512hz_ready_lenfixed.csv")
OUT_DIR = MANIFEST.parent / "manual_vs_auto_training_segments"
OUT_DIR.mkdir(exist_ok=True)

EPOCH_SEC = 5
SMOOTH_MIN = 5
SMOOTH_EPOCHS = int(SMOOTH_MIN * 60 / EPOCH_SEC)

STATE_ORDER = ["Awake", "NREM", "REM", "Undefined"]
STATE_TO_Y = {s: i for i, s in enumerate(STATE_ORDER)}

def load_stage_duration(path, n_epochs=None):
    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

    states = []

    if lines[0].startswith("*Duration"):
        prev_end = 0
        for line in lines[2:]:
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue

            label = " ".join(parts[:-1])
            end_sec = float(parts[-1])

            start_epoch = int(round(prev_end / EPOCH_SEC))
            end_epoch = int(round(end_sec / EPOCH_SEC))

            states.extend([label] * max(0, end_epoch - start_epoch))
            prev_end = end_sec
    else:
        states = lines

    states = [normalize_state(s) for s in states]

    if n_epochs is not None:
        if len(states) < n_epochs:
            states.extend(["Undefined"] * (n_epochs - len(states)))
        elif len(states) > n_epochs:
            states = states[:n_epochs]

    return np.array(states)

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
        "NonREM": "NREM",

        "REM": "REM",
        "Rem": "REM",
        "PS": "REM",
        "Paradoxical Sleep": "REM",

        "Undefined": "Undefined",
        "undefined": "Undefined",
        "ND": "Undefined",
        "NaN": "Undefined",
        "nan": "Undefined",
    }

    return mapping.get(s, s)

def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    arrays = [np.asarray(z[state], dtype=float) for state in state_names]
    probs = np.vstack(arrays).T
    return probs, state_names

def rolling_mean(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.vstack([
        np.convolve(x[:, i], kernel, mode="same")
        for i in range(x.shape[1])
    ]).T

def states_to_y(states):
    return np.array([STATE_TO_Y.get(s, STATE_TO_Y["Undefined"]) for s in states])

df = pd.read_csv(MANIFEST)

summary_rows = []

for i, row in df.iterrows():
    prob_path = Path(row["file_path_state_probabilities"])
    manual_path = Path(row["file_path_manual_state_annotation"])
    auto_path = Path(row["file_path_automated_state_annotation"])

    probs, prob_state_names = load_probabilities(prob_path)
    n_epochs = probs.shape[0]

    manual = load_stage_duration(manual_path, n_epochs=n_epochs)
    auto = load_stage_duration(auto_path, n_epochs=n_epochs)

    auto_from_prob = np.array(prob_state_names)[np.argmax(probs, axis=1)]

    # If automated annotation file parsing gives weird result, fall back to probabilities.
    if len(set(auto)) <= 1 and len(set(auto_from_prob)) > 1:
        auto = auto_from_prob

    t_hours = np.arange(n_epochs) * EPOCH_SEC / 3600
    smooth_probs = rolling_mean(probs, SMOOTH_EPOCHS)

    accuracy = np.mean(manual == auto) * 100

    for state in STATE_ORDER:
        summary_rows.append({
            "row": i,
            "recording_name": row["recording_name"],
            "mouse_id": row["mouse_id"],
            "segment_id": row["segment_id"],
            "state": state,
            "manual_pct": np.mean(manual == state) * 100,
            "auto_pct": np.mean(auto == state) * 100,
            "accuracy_pct": accuracy,
        })

    fig, axes = plt.subplots(
        3, 1,
        figsize=(16, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 2.5]}
    )

    axes[0].step(t_hours, states_to_y(manual), where="post", linewidth=0.8)
    axes[0].set_yticks(range(len(STATE_ORDER)))
    axes[0].set_yticklabels(STATE_ORDER)
    axes[0].set_ylabel("Manual")

    axes[1].step(t_hours, states_to_y(auto), where="post", linewidth=0.8)
    axes[1].set_yticks(range(len(STATE_ORDER)))
    axes[1].set_yticklabels(STATE_ORDER)
    axes[1].set_ylabel("Somnotate")

    for j, state in enumerate(prob_state_names):
        axes[2].plot(t_hours, smooth_probs[:, j], label=state, linewidth=1.2)

    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel(f"{SMOOTH_MIN}-min smoothed probability")
    axes[2].set_xlabel("Time from segment start (hours)")
    axes[2].legend(loc="upper right")

    fig.suptitle(
        f"Manual vs Somnotate | row {i} | mouse {row['mouse_id']} | "
        f"segment {row['segment_id']} | accuracy {accuracy:.1f}%\n"
        f"{row['recording_name']}",
        y=0.98
    )

    plt.tight_layout()

    out = OUT_DIR / f"row{i}_mouse{row['mouse_id']}_segment{row['segment_id']}_manual_vs_auto.png"
    plt.savefig(out, dpi=180)
    plt.close()

    print("Wrote:", out)

summary = pd.DataFrame(summary_rows)
summary_out = OUT_DIR / "manual_vs_auto_state_summary.csv"
summary.to_csv(summary_out, index=False)

print("\nWrote summary:", summary_out)
print("\nOpen folder:")
print(OUT_DIR)
