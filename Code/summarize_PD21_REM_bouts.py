from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prefer final WT-model PD inference manifest
FINAL_COMPLETED = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz_completed.csv")
FINAL_RAW = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz.csv")

# Fallback to older pilot manifest
PILOT = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv")

if FINAL_COMPLETED.exists():
    MANIFEST = FINAL_COMPLETED
elif FINAL_RAW.exists():
    MANIFEST = FINAL_RAW
else:
    MANIFEST = PILOT

OUT_DIR = MANIFEST.parent / "PD21_REM_bout_QC"
OUT_DIR.mkdir(exist_ok=True)

EPOCH_SEC = 5
REM_PROB_THRESHOLD = 0.8
MIN_STABLE_REM_SEC = 30
TRANSITION_EXCLUSION_SEC = 30

def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[s], dtype=float) for s in state_names]).T
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)

    if "REM" not in state_names:
        raise ValueError(f"No REM key in {path}; keys={state_names}")

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

df = pd.read_csv(MANIFEST)

pd21 = df[
    (df["group"] == "PD")
    & (df["week"] == 21)
].copy()

print("Using manifest:", MANIFEST)
print("PD week 21 rows:", len(pd21))

segment_rows = []
bout_rows = []

for i, row in pd21.iterrows():
    prob_path = Path(row["file_path_state_probabilities"])

    if not prob_path.exists():
        print("Missing probability file:", prob_path)
        continue

    probs, state_names, pred, confidence, rem_prob = load_probabilities(prob_path)

    n_epochs = len(pred)
    duration_h = n_epochs * EPOCH_SEC / 3600

    rem_bouts = get_bouts(pred, "REM")
    transition_epochs = get_transition_epochs(pred)
    dist_to_transition_epochs = distance_to_nearest_transition(n_epochs, transition_epochs)

    high_conf_rem_epoch = (
        (pred == "REM")
        & (rem_prob >= REM_PROB_THRESHOLD)
        & (dist_to_transition_epochs * EPOCH_SEC >= TRANSITION_EXCLUSION_SEC)
    )

    segment_rows.append({
        "recording_name": row["recording_name"],
        "mouse_id": row["mouse_id"],
        "week": row["week"],
        "segment_id": row["segment_id"],
        "duration_h": duration_h,
        "n_epochs": n_epochs,
        "pct_pred_REM": np.mean(pred == "REM") * 100,
        "total_pred_REM_min": np.sum(pred == "REM") * EPOCH_SEC / 60,
        "total_high_conf_stable_REM_min": np.sum(high_conf_rem_epoch) * EPOCH_SEC / 60,
        "n_pred_REM_bouts": len(rem_bouts),
        "transitions_per_hour": len(transition_epochs) / duration_h if duration_h else np.nan,
        "mean_confidence": np.mean(confidence),
        "mean_REM_prob": np.mean(rem_prob),
        "pct_low_confidence_lt_0_8": np.mean(confidence < 0.8) * 100,
        "probability_path": str(prob_path),
    })

    for bout_id, (start, end) in enumerate(rem_bouts):
        dur_epochs = end - start + 1
        dur_sec = dur_epochs * EPOCH_SEC

        bout_rem_prob = rem_prob[start:end+1]
        bout_conf = confidence[start:end+1]
        bout_dist_sec = dist_to_transition_epochs[start:end+1] * EPOCH_SEC

        is_high_conf = np.mean(bout_rem_prob >= REM_PROB_THRESHOLD) >= 0.8
        is_far_from_transition = np.min(bout_dist_sec) >= TRANSITION_EXCLUSION_SEC
        is_long_enough = dur_sec >= MIN_STABLE_REM_SEC

        stable = bool(is_high_conf and is_far_from_transition and is_long_enough)

        bout_rows.append({
            "recording_name": row["recording_name"],
            "mouse_id": row["mouse_id"],
            "week": row["week"],
            "segment_id": row["segment_id"],
            "bout_id": bout_id,
            "start_epoch": start,
            "end_epoch": end,
            "start_sec": start * EPOCH_SEC,
            "end_sec": (end + 1) * EPOCH_SEC,
            "start_min": start * EPOCH_SEC / 60,
            "end_min": (end + 1) * EPOCH_SEC / 60,
            "duration_sec": dur_sec,
            "duration_min": dur_sec / 60,
            "mean_REM_prob": np.mean(bout_rem_prob),
            "min_REM_prob": np.min(bout_rem_prob),
            "max_REM_prob": np.max(bout_rem_prob),
            "mean_confidence": np.mean(bout_conf),
            "min_distance_to_transition_sec": np.min(bout_dist_sec),
            "short_REM_lt_30s": dur_sec < 30,
            "stable_high_conf_REM": stable,
            "probability_path": str(prob_path),
        })

segment_summary = pd.DataFrame(segment_rows)
bout_summary = pd.DataFrame(bout_rows)

segment_summary.to_csv(OUT_DIR / "PD21_segment_REM_summary.csv", index=False)
bout_summary.to_csv(OUT_DIR / "PD21_REM_bouts.csv", index=False)

if len(segment_summary):
    mouse_summary = segment_summary.groupby("mouse_id", as_index=False).agg(
        n_segments=("segment_id", "count"),
        total_duration_h=("duration_h", "sum"),
        mean_pct_pred_REM=("pct_pred_REM", "mean"),
        total_pred_REM_min=("total_pred_REM_min", "sum"),
        total_high_conf_stable_REM_min=("total_high_conf_stable_REM_min", "sum"),
        mean_n_pred_REM_bouts=("n_pred_REM_bouts", "mean"),
        mean_transitions_per_hour=("transitions_per_hour", "mean"),
        mean_confidence=("mean_confidence", "mean"),
    )
else:
    mouse_summary = pd.DataFrame()

if len(bout_summary):
    bout_mouse_summary = bout_summary.groupby("mouse_id", as_index=False).agg(
        n_REM_bouts=("bout_id", "count"),
        n_short_REM_lt_30s=("short_REM_lt_30s", "sum"),
        n_stable_high_conf_REM=("stable_high_conf_REM", "sum"),
        mean_REM_bout_duration_min=("duration_min", "mean"),
        median_REM_bout_duration_min=("duration_min", "median"),
        mean_bout_REM_prob=("mean_REM_prob", "mean"),
    )

    mouse_summary = mouse_summary.merge(bout_mouse_summary, on="mouse_id", how="left")

mouse_summary.to_csv(OUT_DIR / "PD21_mouse_REM_summary.csv", index=False)

print("\nWrote:")
print(OUT_DIR / "PD21_segment_REM_summary.csv")
print(OUT_DIR / "PD21_REM_bouts.csv")
print(OUT_DIR / "PD21_mouse_REM_summary.csv")

print("\nPD21 mouse summary:")
print(mouse_summary.to_string(index=False))

# ---------------- PLOTS ----------------
if len(mouse_summary):
    def barplot(col, ylabel, title, filename):
        plt.figure(figsize=(7, 4.5))
        x = np.arange(len(mouse_summary))
        plt.bar(x, mouse_summary[col].values)
        plt.xticks(x, ["M" + str(m) for m in mouse_summary["mouse_id"]])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out = OUT_DIR / filename
        plt.savefig(out, dpi=180)
        plt.close()
        print("Wrote:", out)

    barplot(
        "total_pred_REM_min",
        "Predicted REM minutes",
        "PD week 21: total predicted REM per mouse",
        "PD21_total_pred_REM_min_by_mouse.png",
    )

    barplot(
        "total_high_conf_stable_REM_min",
        "Stable high-confidence REM minutes",
        "PD week 21: stable high-confidence REM per mouse",
        "PD21_stable_high_conf_REM_min_by_mouse.png",
    )

    barplot(
        "n_REM_bouts",
        "Number of REM bouts",
        "PD week 21: predicted REM bout count",
        "PD21_REM_bout_count_by_mouse.png",
    )

    barplot(
        "n_short_REM_lt_30s",
        "REM bouts < 30 s",
        "PD week 21: short predicted REM bouts",
        "PD21_short_REM_bouts_by_mouse.png",
    )

if len(bout_summary):
    plt.figure(figsize=(7, 4.5))
    plt.hist(bout_summary["duration_sec"], bins=30)
    plt.axvline(30, linestyle="--", linewidth=1, label="30 s")
    plt.axvline(60, linestyle="--", linewidth=1, label="60 s")
    plt.xlabel("REM bout duration (s)")
    plt.ylabel("Count")
    plt.title("PD week 21 predicted REM bout duration distribution")
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / "PD21_REM_bout_duration_histogram.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(bout_summary["duration_sec"], bout_summary["mean_REM_prob"], s=20)
    plt.axhline(REM_PROB_THRESHOLD, linestyle="--", linewidth=1)
    plt.axvline(30, linestyle="--", linewidth=1)
    plt.xlabel("REM bout duration (s)")
    plt.ylabel("Mean REM probability")
    plt.title("PD week 21 REM bout duration vs REM probability")
    plt.tight_layout()
    out = OUT_DIR / "PD21_REM_bout_duration_vs_prob.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

print("\nDone.")
print("Open folder:")
print(OUT_DIR)
