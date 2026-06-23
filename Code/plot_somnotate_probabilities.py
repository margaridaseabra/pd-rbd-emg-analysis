from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MANIFEST = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv")

OUT_DIR = MANIFEST.parent / "plots_512hz"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(MANIFEST)

print("Using manifest:", MANIFEST)
print("Rows:", len(df))
print("Output folder:", OUT_DIR)

def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    print("\nProbability file:", path)
    print("NPZ keys:", z.files)

    # Somnotate stores one 1D array per state: Awake, NREM, REM
    state_names = list(z.files)
    arrays = [z[state] for state in state_names]

    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Probability arrays have different lengths: {dict(zip(state_names, lengths))}")

    probs = np.vstack(arrays).T  # shape: n_epochs x n_states

    print("Probability shape:", probs.shape)
    print("State names:", state_names)

    return probs, state_names

def choose_examples(df):
    examples = []

    candidates = [
        ("PD_week2",  "PD", 2),
        ("WT_week2",  "WT", 2),
        ("PD_week21", "PD", 21),
        ("WT_week21", "WT", 21),
    ]

    for label, group, week in candidates:
        sub = df[(df["group"] == group) & (df["week"] == week)].copy()
        if len(sub):
            examples.append((label, sub.iloc[0]))

    return examples

examples = choose_examples(df)

if not examples:
    raise SystemExit("No examples found in manifest.")

for label, row in examples:
    prob_path = Path(row["file_path_state_probabilities"])

    probs, state_names = load_probabilities(prob_path)

    t_hours = np.arange(probs.shape[0]) * 5 / 3600

    plt.figure(figsize=(14, 5))

    for i, state_name in enumerate(state_names):
        plt.plot(t_hours, probs[:, i], label=state_name, linewidth=1)

    title = (
        f"{label} | mouse {row['mouse_id']} | segment {row['segment_id']} | "
        f"{row['recording_name']}"
    )

    plt.title(title)
    plt.xlabel("Time from segment start (hours)")
    plt.ylabel("State probability")
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="upper right")
    plt.tight_layout()

    safe_name = f"{label}_mouse{row['mouse_id']}_segment{row['segment_id']}_probabilities.png"
    out_path = OUT_DIR / safe_name
    plt.savefig(out_path, dpi=180)
    plt.close()

    print("Wrote:", out_path)

print("\nDone.")
print("Open this folder:")
print(OUT_DIR)
