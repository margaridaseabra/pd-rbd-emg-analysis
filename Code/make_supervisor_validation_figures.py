from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data")
OUT_DIR = Path.home() / "Desktop" / "somnotate_supervisor_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5
SMOOTH_MIN = 5
SMOOTH_EPOCHS = int(SMOOTH_MIN * 60 / EPOCH_SEC)

STATE_ORDER = ["Awake", "NREM", "REM", "Undefined"]
STATE_TO_Y = {s: i for i, s in enumerate(STATE_ORDER)}

# Five WT week 2 internal-validation segments from your current run
all_validation_segments = [
    {
        "label": "M1 seg0",
        "recording_name": "LC_PD_wk2_M1_default",
        "mouse_id": 1,
        "segment_id": 0,
        "accuracy": 95.2,
        "folder": BASE / "LC_PD_wk2_M1_default" / "segment_00",
    },
    {
        "label": "M7 seg0",
        "recording_name": "20231218_LC_PD_2wk_Mouse7_2023-12-19_09-10-35-529",
        "mouse_id": 7,
        "segment_id": 0,
        "accuracy": 89.4,
        "folder": BASE / "20231218_LC_PD_2wk_Mouse7_2023-12-19_09-10-35-529" / "segment_00",
    },
    {
        "label": "M7 seg1",
        "recording_name": "20231218_LC_PD_2wk_Mouse7_2023-12-19_09-10-35-529",
        "mouse_id": 7,
        "segment_id": 1,
        "accuracy": 95.5,
        "folder": BASE / "20231218_LC_PD_2wk_Mouse7_2023-12-19_09-10-35-529" / "segment_01",
    },
    {
        "label": "M10 seg2",
        "recording_name": "LC_PD_wk2_M10_default",
        "mouse_id": 10,
        "segment_id": 2,
        "accuracy": 90.9,
        "folder": BASE / "LC_PD_wk2_M10_default" / "segment_02",
    },
    {
        "label": "M11 seg0",
        "recording_name": "20231218_LC_PD_2wk_Mouse11_2023-12-19_09-10-35-529",
        "mouse_id": 11,
        "segment_id": 0,
        "accuracy": 94.8,
        "folder": BASE / "20231218_LC_PD_2wk_Mouse11_2023-12-19_09-10-35-529" / "segment_00",
    },
]

# Representative examples for your supervisor slides
selected_examples = [
    {**all_validation_segments[2], "example_type": "Good example"},
    {**all_validation_segments[1], "example_type": "Lower-performing example"},
    {**all_validation_segments[4], "example_type": "REM-rich example"},
]

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

def find_disagreement_windows(manual, auto, window_epochs=240, max_windows=3):
    disagreement = manual != auto
    idx = np.where(disagreement)[0]

    if len(idx) == 0:
        return []

    windows = []
    used = np.zeros(len(manual), dtype=bool)

    for center in idx:
        if used[center]:
            continue

        start = max(0, center - window_epochs // 2)
        end = min(len(manual), center + window_epochs // 2)

        windows.append((start, end))
        used[start:end] = True

        if len(windows) >= max_windows:
            break

    return windows

def plot_accuracy_bar():
    labels = [x["label"] for x in all_validation_segments]
    accuracies = [x["accuracy"] for x in all_validation_segments]
    mean_acc = np.mean(accuracies)

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(labels))
    plt.bar(x, accuracies)
    plt.axhline(mean_acc, linestyle="--", linewidth=1)
    plt.text(len(labels) - 0.5, mean_acc + 0.5, f"Mean = {mean_acc:.2f}%", ha="right")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(80, 100)
    plt.title("Somnotate internal validation on WT week 2 labeled segments")
    plt.tight_layout()

    out = OUT_DIR / "01_validation_accuracy_bar.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

def make_example_plot(example):
    folder = example["folder"]

    manual_path = folder / "somnotate_annotation_lenfixed_512hz.tsv"
    auto_path = folder / "somnotate_automated.tsv"
    prob_path = folder / "somnotate_state_probabilities.npz"

    for p in [manual_path, auto_path, prob_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    probs, prob_state_names = load_probabilities(prob_path)
    n_epochs = probs.shape[0]
    t_hours = np.arange(n_epochs) * EPOCH_SEC / 3600

    manual = load_stage_duration(manual_path, n_epochs)
    auto = load_stage_duration(auto_path, n_epochs)

    # If automated file parses badly, fall back to argmax probabilities.
    auto_from_prob = np.array(prob_state_names)[np.argmax(probs, axis=1)]
    if len(set(auto)) <= 1 and len(set(auto_from_prob)) > 1:
        auto = auto_from_prob

    smooth_probs = rolling_mean(probs, SMOOTH_EPOCHS)

    disagreement = manual != auto
    agreement = np.mean(~disagreement) * 100

    # Whole-segment plot
    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 2.5, 0.55]},
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
    axes[2].legend(loc="upper right")

    axes[3].fill_between(t_hours, 0, disagreement.astype(int), step="post")
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["agree", "disagree"])
    axes[3].set_xlabel("Time from segment start (hours)")
    axes[3].set_ylabel("Mismatch")

    fig.suptitle(
        f"{example['example_type']} | {example['label']} | "
        f"accuracy from Somnotate test: {example['accuracy']:.1f}% | "
        f"agreement in plot: {agreement:.1f}%\n"
        f"{example['recording_name']}",
        y=0.99,
    )

    plt.tight_layout()

    safe = example["example_type"].lower().replace(" ", "_").replace("-", "_")
    out = OUT_DIR / f"02_{safe}_{example['label'].replace(' ', '_')}_manual_vs_auto.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

    # Disagreement zooms
    windows = find_disagreement_windows(manual, auto)

    for k, (start, end) in enumerate(windows, start=1):
        tt = t_hours[start:end]

        fig, axes = plt.subplots(
            3, 1,
            figsize=(13, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 2]},
        )

        axes[0].step(tt, states_to_y(manual[start:end]), where="post", linewidth=1.2)
        axes[0].set_yticks(range(len(STATE_ORDER)))
        axes[0].set_yticklabels(STATE_ORDER)
        axes[0].set_ylabel("Manual")

        axes[1].step(tt, states_to_y(auto[start:end]), where="post", linewidth=1.2)
        axes[1].set_yticks(range(len(STATE_ORDER)))
        axes[1].set_yticklabels(STATE_ORDER)
        axes[1].set_ylabel("Somnotate")

        for j, state in enumerate(prob_state_names):
            axes[2].plot(tt, probs[start:end, j], label=state, linewidth=1.1)

        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_ylabel("Raw state probability")
        axes[2].set_xlabel("Time from segment start (hours)")
        axes[2].legend(loc="upper right")

        fig.suptitle(
            f"Disagreement window {k} | {example['example_type']} | {example['label']}",
            y=0.98,
        )

        plt.tight_layout()

        out = OUT_DIR / f"03_{safe}_{example['label'].replace(' ', '_')}_disagreement_zoom{k}.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print("Wrote:", out)

    # Summary row
    state_summary = {}
    for state in ["Awake", "NREM", "REM", "Undefined"]:
        state_summary[f"manual_pct_{state}"] = np.mean(manual == state) * 100
        state_summary[f"auto_pct_{state}"] = np.mean(auto == state) * 100

    return {
        "example_type": example["example_type"],
        "label": example["label"],
        "recording_name": example["recording_name"],
        "mouse_id": example["mouse_id"],
        "segment_id": example["segment_id"],
        "somnotate_test_accuracy_pct": example["accuracy"],
        "agreement_recomputed_pct": agreement,
        "n_epochs": n_epochs,
        "duration_hours": n_epochs * EPOCH_SEC / 3600,
        "n_disagreement_epochs": int(np.sum(disagreement)),
        "disagreement_pct": np.mean(disagreement) * 100,
        **state_summary,
    }

def main():
    print("Writing figures to:")
    print(OUT_DIR)
    print()

    plot_accuracy_bar()

    rows = []
    for example in selected_examples:
        rows.append(make_example_plot(example))

    summary = pd.DataFrame(rows)
    out = OUT_DIR / "validation_examples_summary.csv"
    summary.to_csv(out, index=False)
    print("Wrote:", out)

    print("\nDone.")
    print("Open this folder:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()
