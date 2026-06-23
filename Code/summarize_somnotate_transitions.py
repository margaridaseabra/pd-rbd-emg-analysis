from pathlib import Path
import numpy as np
import pandas as pd

MANIFEST = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv")
OUT_DIR = MANIFEST.parent / "somnotate_transition_summary"
OUT_DIR.mkdir(exist_ok=True)

EPOCH_SEC = 5

def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[s], dtype=float) for s in state_names]).T
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)
    return probs, state_names, pred, confidence

def get_bouts(states):
    states = np.asarray(states)
    starts = np.where(np.r_[True, states[1:] != states[:-1]])[0]
    ends = np.r_[starts[1:] - 1, len(states) - 1]
    return starts, ends, states[starts]

df = pd.read_csv(MANIFEST)

segment_rows = []
transition_rows = []
bout_rows = []

for idx, row in df.iterrows():
    prob_path = Path(row["file_path_state_probabilities"])
    if not prob_path.exists():
        print("Missing:", prob_path)
        continue

    probs, state_names, pred, confidence = load_probabilities(prob_path)
    n_epochs = len(pred)
    duration_hours = n_epochs * EPOCH_SEC / 3600

    starts, ends, bout_states = get_bouts(pred)

    # Per-segment state occupancy and confidence
    seg = {
        "row": idx,
        "recording_name": row["recording_name"],
        "mouse_id": row["mouse_id"],
        "group": row["group"],
        "week": row["week"],
        "segment_id": row["segment_id"],
        "recommended_use": row["recommended_use"],
        "duration_hours": duration_hours,
        "mean_confidence": confidence.mean(),
        "pct_low_confidence_lt_0_8": np.mean(confidence < 0.8) * 100,
        "n_transitions": max(0, len(starts) - 1),
        "transitions_per_hour": max(0, len(starts) - 1) / duration_hours if duration_hours > 0 else np.nan,
    }

    for state in state_names:
        seg[f"pct_{state}"] = np.mean(pred == state) * 100
        seg[f"minutes_{state}"] = np.sum(pred == state) * EPOCH_SEC / 60

    segment_rows.append(seg)

    # Transition events
    for j in range(1, len(starts)):
        from_state = bout_states[j - 1]
        to_state = bout_states[j]
        transition_epoch = starts[j]
        transition_time_sec = transition_epoch * EPOCH_SEC

        transition_rows.append({
            "row": idx,
            "recording_name": row["recording_name"],
            "mouse_id": row["mouse_id"],
            "group": row["group"],
            "week": row["week"],
            "segment_id": row["segment_id"],
            "from_state": from_state,
            "to_state": to_state,
            "transition_epoch": transition_epoch,
            "transition_time_sec": transition_time_sec,
            "transition_time_min": transition_time_sec / 60,
            "confidence_at_transition": confidence[transition_epoch],
        })

    # Bouts
    for start, end, state in zip(starts, ends, bout_states):
        duration_epochs = end - start + 1
        bout_rows.append({
            "row": idx,
            "recording_name": row["recording_name"],
            "mouse_id": row["mouse_id"],
            "group": row["group"],
            "week": row["week"],
            "segment_id": row["segment_id"],
            "state": state,
            "start_time_sec": start * EPOCH_SEC,
            "end_time_sec": (end + 1) * EPOCH_SEC,
            "duration_sec": duration_epochs * EPOCH_SEC,
            "duration_min": duration_epochs * EPOCH_SEC / 60,
            "mean_confidence": confidence[start:end+1].mean(),
        })

segment_summary = pd.DataFrame(segment_rows)
transition_summary = pd.DataFrame(transition_rows)
bout_summary = pd.DataFrame(bout_rows)

segment_summary.to_csv(OUT_DIR / "segment_state_probability_summary.csv", index=False)
transition_summary.to_csv(OUT_DIR / "state_transitions.csv", index=False)
bout_summary.to_csv(OUT_DIR / "state_bouts.csv", index=False)

# Transition matrix
if len(transition_summary):
    transition_matrix = pd.crosstab(
        transition_summary["from_state"],
        transition_summary["to_state"]
    )
    transition_matrix.to_csv(OUT_DIR / "transition_matrix.csv")

# REM-specific summary
rem_bouts = bout_summary[bout_summary["state"] == "REM"].copy()

if len(rem_bouts):
    rem_summary = rem_bouts.groupby(
        ["group", "week", "mouse_id", "recording_name", "segment_id"],
        as_index=False
    ).agg(
        n_rem_bouts=("duration_min", "count"),
        total_rem_min=("duration_min", "sum"),
        mean_rem_bout_min=("duration_min", "mean"),
        median_rem_bout_min=("duration_min", "median"),
        short_rem_bouts_lt_30s=("duration_sec", lambda x: np.sum(x < 30)),
        mean_rem_confidence=("mean_confidence", "mean"),
    )

    rem_summary.to_csv(OUT_DIR / "rem_bout_summary.csv", index=False)

print("Wrote summaries to:")
print(OUT_DIR)

print("\nSegment summary:")
print(segment_summary[["group", "week", "mouse_id", "segment_id", "pct_Awake", "pct_NREM", "pct_REM", "mean_confidence", "transitions_per_hour"]].head().to_string(index=False))

if len(transition_summary):
    print("\nTransition counts:")
    print(pd.crosstab(transition_summary["from_state"], transition_summary["to_state"]))

if len(rem_bouts):
    print("\nREM summary preview:")
    print(rem_summary.head().to_string(index=False))
