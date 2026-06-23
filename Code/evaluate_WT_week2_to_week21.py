from pathlib import Path
import numpy as np
import pandas as pd

EPOCH_SEC = 5
STATES = ["Awake", "NREM", "REM"]

MANIFEST = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/WT_week2_to_week21_transfer/WT_week21_test_WTwk2model.csv")
OUT_DIR = MANIFEST.parent / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)

def normalize_state(s):
    s = str(s).strip()
    mapping = {
        "Wake": "Awake", "WK": "Awake", "W": "Awake", "wake": "Awake", "AWAKE": "Awake",
        "SWS": "NREM", "NREM": "NREM", "Nrem": "NREM",
        "PS": "REM", "REM": "REM", "Rem": "REM",
        "TR": "Undefined", "ND": "Undefined", "Undefined": "Undefined", "nan": "Undefined",
    }
    return mapping.get(s, s)

def load_stage_duration(path, n_epochs=None):
    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    states = []

    if lines and lines[0].startswith("*Duration"):
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

df = pd.read_csv(MANIFEST)

rows = []
all_manual = []
all_auto = []

for _, row in df.iterrows():
    manual_path = Path(row["file_path_manual_state_annotation"])
    auto_path = Path(row["file_path_automated_state_annotation"])

    if not manual_path.exists() or not auto_path.exists():
        print("Skipping missing file:", row["recording_name"], row["segment_id"])
        continue

    auto = load_stage_duration(auto_path)
    manual = load_stage_duration(manual_path, n_epochs=len(auto))

    valid = np.isin(manual, STATES)
    manual_valid = manual[valid]
    auto_valid = auto[valid]

    acc = np.mean(manual_valid == auto_valid) * 100

    rows.append({
        "recording_name": row["recording_name"],
        "mouse_id": row["mouse_id"],
        "segment_id": row["segment_id"],
        "accuracy_pct": acc,
        "manual_REM_pct": np.mean(manual_valid == "REM") * 100,
        "auto_REM_pct": np.mean(auto_valid == "REM") * 100,
        "n_valid_epochs": len(manual_valid),
    })

    all_manual.extend(manual_valid)
    all_auto.extend(auto_valid)

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "WT_week2_to_week21_segment_metrics.csv", index=False)

all_manual = np.array(all_manual)
all_auto = np.array(all_auto)

overall = np.mean(all_manual == all_auto) * 100

state_rows = []
for state in STATES:
    true_state = all_manual == state
    pred_state = all_auto == state
    tp = np.sum(true_state & pred_state)
    recall = tp / np.sum(true_state) * 100 if np.sum(true_state) else np.nan
    precision = tp / np.sum(pred_state) * 100 if np.sum(pred_state) else np.nan
    state_rows.append({
        "state": state,
        "recall_pct": recall,
        "precision_pct": precision,
        "manual_epochs": int(np.sum(true_state)),
        "auto_epochs": int(np.sum(pred_state)),
    })

state_metrics = pd.DataFrame(state_rows)
state_metrics.to_csv(OUT_DIR / "WT_week2_to_week21_state_metrics.csv", index=False)

print("\nWT week 2 → WT week 21 evaluation")
print("Overall accuracy:", f"{overall:.2f}%")
print()
print("Segment metrics:")
print(summary.to_string(index=False))
print()
print("State metrics:")
print(state_metrics.to_string(index=False))
print()
print("Output folder:")
print(OUT_DIR)
