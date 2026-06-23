from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FINAL_COMPLETED = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz_completed.csv")
FINAL_RAW = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz.csv")

if FINAL_COMPLETED.exists():
    MANIFEST = FINAL_COMPLETED
else:
    MANIFEST = FINAL_RAW

OUT_DIR = MANIFEST.parent / "PD_transfer_evaluation_finalWT"
OUT_DIR.mkdir(exist_ok=True)

EPOCH_SEC = 5
STATES = ["Awake", "NREM", "REM"]

def normalize_state(s):
    s = str(s).strip()
    mapping = {
        "Wake": "Awake", "W": "Awake", "AWAKE": "Awake", "wake": "Awake", "awake": "Awake",
        "NREM": "NREM", "Nrem": "NREM", "SWS": "NREM", "sws": "NREM", "NonREM": "NREM",
        "REM": "REM", "Rem": "REM", "PS": "REM", "ps": "REM", "Paradoxical Sleep": "REM",
        "Undefined": "Undefined", "undefined": "Undefined", "ND": "Undefined", "nan": "Undefined", "NaN": "Undefined",
        "TR": "Undefined",
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

def load_stage_duration(path, n_epochs=None):
    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

    states = []

    if not lines:
        states = []

    elif lines[0].startswith("*Duration"):
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

def load_prob_argmax(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[s], dtype=float) for s in state_names]).T
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)
    return pred, confidence, probs, state_names

df = pd.read_csv(MANIFEST)

# Keep only PD rows with manual labels and final model outputs
df["manual_exists"] = df["file_path_manual_state_annotation"].map(lambda p: choose_manual_annotation_path(p).exists() if isinstance(p, str) else False)
df["auto_exists"] = df["file_path_automated_state_annotation"].map(lambda p: Path(p).exists() if isinstance(p, str) else False)
df["prob_exists"] = df["file_path_state_probabilities"].map(lambda p: Path(p).exists() if isinstance(p, str) else False)

pd_eval = df[
    (df["group"] == "PD")
    & (df["week"].isin([2, 21]))
    & df["manual_exists"]
    & df["auto_exists"]
    & df["prob_exists"]
].copy()

if "pct_scored" in pd_eval.columns:
    pd_eval = pd_eval[pd_eval["pct_scored"] >= 0.90].copy()

print("Using manifest:", MANIFEST)
print("PD rows for evaluation:", len(pd_eval))
print(pd_eval[["recording_name", "mouse_id", "week", "segment_id", "pct_scored", "label_quality"]].to_string(index=False))

segment_rows = []
all_manual = []
all_auto = []
all_confidence = []
failed_rows = []

for _, row in pd_eval.iterrows():
    manual_path = choose_manual_annotation_path(row["file_path_manual_state_annotation"])
    auto_path = Path(row["file_path_automated_state_annotation"])
    prob_path = Path(row["file_path_state_probabilities"])

    try:
        pred_from_prob, confidence, probs, prob_states = load_prob_argmax(prob_path)
        n_epochs = len(pred_from_prob)

        manual = load_stage_duration(manual_path, n_epochs=n_epochs)
        auto = load_stage_duration(auto_path, n_epochs=n_epochs)

        # Fallback to probability argmax if automated annotation file parses strangely
        if len(set(auto)) <= 1 and len(set(pred_from_prob)) > 1:
            auto = pred_from_prob

    except Exception as e:
        failed_rows.append({
            "recording_name": row["recording_name"],
            "mouse_id": row["mouse_id"],
            "week": row["week"],
            "segment_id": row["segment_id"],
            "error": repr(e),
        })
        continue

    valid = np.isin(manual, STATES)
    manual_valid = manual[valid]
    auto_valid = auto[valid]
    confidence_valid = confidence[valid]

    accuracy = np.mean(manual_valid == auto_valid) * 100 if len(manual_valid) else np.nan

    out = {
        "recording_name": row["recording_name"],
        "mouse_id": row["mouse_id"],
        "week": row["week"],
        "segment_id": row["segment_id"],
        "recommended_use": row.get("recommended_use", ""),
        "label_quality": row.get("label_quality", ""),
        "n_epochs_total": n_epochs,
        "n_valid_manual_epochs": len(manual_valid),
        "accuracy_pct": accuracy,
        "mean_confidence": np.mean(confidence_valid) if len(confidence_valid) else np.nan,
        "pct_low_confidence_lt_0_8": np.mean(confidence_valid < 0.8) * 100 if len(confidence_valid) else np.nan,
    }

    for state in STATES:
        out[f"manual_pct_{state}"] = np.mean(manual_valid == state) * 100 if len(manual_valid) else np.nan
        out[f"auto_pct_{state}"] = np.mean(auto_valid == state) * 100 if len(auto_valid) else np.nan

    segment_rows.append(out)
    all_manual.extend(manual_valid)
    all_auto.extend(auto_valid)
    all_confidence.extend(confidence_valid)

segment_metrics = pd.DataFrame(segment_rows)
failed = pd.DataFrame(failed_rows)

segment_metrics.to_csv(OUT_DIR / "PD_transfer_segment_metrics.csv", index=False)
failed.to_csv(OUT_DIR / "PD_transfer_failed_rows.csv", index=False)

all_manual = np.array(all_manual)
all_auto = np.array(all_auto)
all_confidence = np.array(all_confidence)

if len(all_manual) == 0:
    raise SystemExit("No valid manually scored PD epochs found.")

overall_accuracy = np.mean(all_manual == all_auto) * 100
mean_confidence = np.mean(all_confidence)

confusion = pd.crosstab(
    pd.Series(all_manual, name="Manual"),
    pd.Series(all_auto, name="Somnotate")
).reindex(index=STATES, columns=STATES, fill_value=0)

confusion.to_csv(OUT_DIR / "PD_transfer_confusion_counts.csv")

confusion_row_pct = confusion.div(confusion.sum(axis=1), axis=0) * 100
confusion_row_pct.to_csv(OUT_DIR / "PD_transfer_confusion_row_percent.csv")

state_rows = []
for state in STATES:
    true_state = all_manual == state
    pred_state = all_auto == state

    tp = np.sum(true_state & pred_state)
    n_true = np.sum(true_state)
    n_pred = np.sum(pred_state)

    recall = tp / n_true * 100 if n_true else np.nan
    precision = tp / n_pred * 100 if n_pred else np.nan

    state_rows.append({
        "state": state,
        "recall_pct": recall,
        "precision_pct": precision,
        "manual_epochs": int(n_true),
        "auto_epochs": int(n_pred),
    })

state_metrics = pd.DataFrame(state_rows)
state_metrics.to_csv(OUT_DIR / "PD_transfer_state_metrics.csv", index=False)

week_summary = segment_metrics.groupby("week", as_index=False).agg(
    n_segments=("segment_id", "count"),
    mean_accuracy_pct=("accuracy_pct", "mean"),
    median_accuracy_pct=("accuracy_pct", "median"),
    mean_confidence=("mean_confidence", "mean"),
    mean_manual_REM_pct=("manual_pct_REM", "mean"),
    mean_auto_REM_pct=("auto_pct_REM", "mean"),
)

mouse_summary = segment_metrics.groupby(["week", "mouse_id"], as_index=False).agg(
    n_segments=("segment_id", "count"),
    mean_accuracy_pct=("accuracy_pct", "mean"),
    mean_confidence=("mean_confidence", "mean"),
    mean_manual_REM_pct=("manual_pct_REM", "mean"),
    mean_auto_REM_pct=("auto_pct_REM", "mean"),
)

week_summary.to_csv(OUT_DIR / "PD_transfer_week_summary.csv", index=False)
mouse_summary.to_csv(OUT_DIR / "PD_transfer_mouse_summary.csv", index=False)

# Plot 1: accuracy by mouse/week
plt.figure(figsize=(9, 4.8))
labels = [f"W{w} M{m}" for w, m in zip(mouse_summary["week"], mouse_summary["mouse_id"])]
x = np.arange(len(labels))
plt.bar(x, mouse_summary["mean_accuracy_pct"].values)
plt.axhline(overall_accuracy, linestyle="--", linewidth=1)
plt.text(len(labels)-0.5, overall_accuracy + 0.5, f"Overall = {overall_accuracy:.1f}%", ha="right")
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Accuracy vs manual (%)")
plt.ylim(0, 100)
plt.title("WT-trained final model applied to PD: agreement with manual scoring")
plt.tight_layout()
plt.savefig(OUT_DIR / "PD_transfer_accuracy_by_mouse_week.png", dpi=180)
plt.close()

# Plot 2: confusion matrix
plt.figure(figsize=(5.5, 5))
plt.imshow(confusion_row_pct.values, vmin=0, vmax=100)
plt.xticks(np.arange(len(STATES)), STATES)
plt.yticks(np.arange(len(STATES)), STATES)
plt.xlabel("Somnotate")
plt.ylabel("Manual")
plt.title("PD transfer confusion matrix (% of manual state)")

for r in range(len(STATES)):
    for c in range(len(STATES)):
        plt.text(c, r, f"{confusion_row_pct.values[r, c]:.1f}", ha="center", va="center")

plt.colorbar(label="%")
plt.tight_layout()
plt.savefig(OUT_DIR / "PD_transfer_confusion_matrix.png", dpi=180)
plt.close()

# Plot 3: manual vs auto REM occupancy
plt.figure(figsize=(9, 4.8))
labels = [f"W{w} M{m} S{s}" for w, m, s in zip(segment_metrics["week"], segment_metrics["mouse_id"], segment_metrics["segment_id"])]
x = np.arange(len(labels))
width = 0.4
plt.bar(x - width/2, segment_metrics["manual_pct_REM"].values, width, label="Manual")
plt.bar(x + width/2, segment_metrics["auto_pct_REM"].values, width, label="Somnotate")
plt.xticks(x, labels, rotation=60, ha="right")
plt.ylabel("% REM")
plt.title("PD: manual vs Somnotate REM occupancy by segment")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "PD_transfer_REM_occupancy_manual_vs_auto.png", dpi=180)
plt.close()

print("\nPD transfer evaluation complete.")
print("Output folder:", OUT_DIR)
print()
print("Overall PD agreement accuracy:", f"{overall_accuracy:.2f}%")
print("Mean confidence:", f"{mean_confidence:.3f}")
print()
print("Week summary:")
print(week_summary.to_string(index=False))
print()
print("State metrics:")
print(state_metrics.to_string(index=False))
print()
print("Failed rows:", len(failed))
