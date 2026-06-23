from pathlib import Path
import numpy as np
import pandas as pd

EPOCH_SEC = 5

BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")

EVENT_TABLE = BASE / "EMG_burst_detection_EEGonly_scored" / "EMG_burst_events_scored.csv"

OUT_DIR = BASE / "supervised_EMG_burst_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EEGONLY_MANIFEST_CANDIDATES = [
    Path.home() / "Desktop/local_sleep_manifests/final_WT_reference_manifests/WT_PD_week2_week21_EEGonly_finalWTref_application.csv",
]

WT_EEGONLY_CANDIDATES = []

FULL_MANIFEST_CANDIDATES = [
    BASE / "final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz_completed.csv",
    BASE / "final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz.csv",
    BASE / "all_segments_inference_512hz_completed.csv",
]

REM_PROB_STABLE = 0.70
WAKE_PROB_MAX_FOR_REM = 0.30
TRANSITION_DISTANCE_SEC = 30


def choose_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load_probabilities(path):
    if not isinstance(path, str) or not path or not Path(path).exists():
        return None, None, None, None

    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[s], dtype=float) for s in state_names]).T
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)

    return probs, state_names, pred, conf


def get_prob(probs, state_names, state):
    if probs is None or state_names is None or state not in state_names:
        return None
    return probs[:, state_names.index(state)]


def transition_distance_sec(pred):
    pred = np.asarray(pred)
    transitions = np.where(pred[1:] != pred[:-1])[0] + 1

    if len(transitions) == 0:
        return np.full(len(pred), np.inf)

    idx = np.arange(len(pred))
    dist = np.full(len(pred), np.inf)

    for t in transitions:
        dist = np.minimum(dist, np.abs(idx - t))

    return dist * EPOCH_SEC


def get_bouts(mask):
    mask = np.asarray(mask).astype(bool)

    if len(mask) == 0:
        return []

    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    ends = np.where(mask & np.r_[~mask[1:], True])[0]

    return list(zip(starts, ends))


def safe_group_week(group, week):
    return f"{group} W{int(week)}"


def load_all_manifests():
    frames = []

    pd_manifest = choose_existing(EEGONLY_MANIFEST_CANDIDATES)
    wt_manifest = choose_existing(WT_EEGONLY_CANDIDATES)

    if pd_manifest is not None:
        pd_df = pd.read_csv(pd_manifest)
        pd_df["manifest_source"] = str(pd_manifest)
        frames.append(pd_df)

    if wt_manifest is not None:
        wt_df = pd.read_csv(wt_manifest)
        wt_df["manifest_source"] = str(wt_manifest)
        frames.append(wt_df)

    if not frames:
        print("No EEG-only manifests found. REM amount/quality metrics will be skipped.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Remove duplicates if a row appears in multiple manifests.
    key_cols = ["recording_name", "mouse_id", "week", "segment_id"]
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

    return df


def classify_event_from_existing_row(row):
    # Use existing event_class if available.
    if "event_class" in row and isinstance(row["event_class"], str) and row["event_class"]:
        return row["event_class"]

    eeg_rem = row.get("mean_EEGonly_P_REM", np.nan)
    eeg_awake = row.get("mean_EEGonly_P_Awake", np.nan)
    eeg_amb = row.get("mean_EEGonly_ambiguity", np.nan)
    dist = row.get("min_EEGonly_distance_to_transition_sec", np.nan)
    delta = row.get("mean_delta_REM_EEGonly_minus_full", np.nan)
    full_rem = row.get("mean_full_P_REM", np.nan)

    if np.isfinite(eeg_awake) and eeg_awake >= 0.70 and eeg_rem < 0.30:
        return "wake_like_movement"

    if (
        np.isfinite(eeg_rem)
        and eeg_rem >= 0.70
        and eeg_awake <= 0.30
        and np.isfinite(delta)
        and delta >= 0.25
        and np.isfinite(full_rem)
        and full_rem < eeg_rem
    ):
        return "candidate_EMG_suppressed_REM"

    if (
        np.isfinite(eeg_rem)
        and eeg_rem >= 0.70
        and eeg_awake <= 0.30
        and np.isfinite(dist)
        and dist >= 30
    ):
        return "stable_EEG_REM_EMG_burst"

    if (
        np.isfinite(eeg_rem)
        and eeg_rem >= 0.50
        and np.isfinite(dist)
        and dist < 30
    ):
        return "REM_transition_EMG_burst"

    if np.isfinite(eeg_amb) and eeg_amb >= 0.25:
        return "mixed_or_ambiguous_state"

    return "other_EMG_burst"


# -------------------------------------------------
# LOAD EVENT TABLE
# -------------------------------------------------
if not EVENT_TABLE.exists():
    raise FileNotFoundError(f"Could not find event table: {EVENT_TABLE}")

events = pd.read_csv(EVENT_TABLE)

if "event_class" not in events.columns:
    events["event_class"] = events.apply(classify_event_from_existing_row, axis=1)
else:
    events["event_class"] = events.apply(classify_event_from_existing_row, axis=1)

events["group_week"] = events.apply(lambda r: safe_group_week(r["group"], r["week"]), axis=1)

events_out = OUT_DIR / "supervised_scored_EMG_events.csv"
events.to_csv(events_out, index=False)

print("Loaded EMG events:", len(events))
print("Wrote cleaned event table:", events_out)

# -------------------------------------------------
# EVENT-LEVEL METRICS PER MOUSE
# -------------------------------------------------
event_count = events.groupby(["group", "week", "mouse_id"]).size().rename("n_emg_events").reset_index()

event_class_counts = (
    pd.crosstab(
        [events["group"], events["week"], events["mouse_id"]],
        events["event_class"]
    )
    .reset_index()
)

event_class_pct = (
    pd.crosstab(
        [events["group"], events["week"], events["mouse_id"]],
        events["event_class"],
        normalize="index"
    ) * 100
).reset_index()

event_class_pct.columns = [
    "group",
    "week",
    "mouse_id",
] + [f"pct_{c}" for c in event_class_pct.columns[3:]]

prob_metrics = events.groupby(["group", "week", "mouse_id"]).agg(
    mean_EMG_z=("mean_EMG_z", "mean"),
    mean_max_EMG_z=("max_EMG_z", "mean"),
    mean_EEGonly_P_REM=("mean_EEGonly_P_REM", "mean"),
    mean_EEGonly_P_Awake=("mean_EEGonly_P_Awake", "mean"),
    mean_full_P_REM=("mean_full_P_REM", "mean"),
    mean_delta_REM=("mean_delta_REM_EEGonly_minus_full", "mean"),
    mean_EEGonly_REM_Wake_balance=("mean_EEGonly_REM_Wake_balance", "mean"),
    mean_EEGonly_ambiguity=("mean_EEGonly_ambiguity", "mean"),
    mean_full_ambiguity=("mean_full_ambiguity", "mean"),
    fraction_near_EEGonly_transition=(
        "min_EEGonly_distance_to_transition_sec",
        lambda x: np.mean(np.asarray(x) < TRANSITION_DISTANCE_SEC) * 100,
    ),
).reset_index()

mouse_event_metrics = (
    event_count
    .merge(event_class_counts, on=["group", "week", "mouse_id"], how="left")
    .merge(event_class_pct, on=["group", "week", "mouse_id"], how="left")
    .merge(prob_metrics, on=["group", "week", "mouse_id"], how="left")
)

mouse_event_out = OUT_DIR / "mouse_level_EMG_event_supervised_metrics.csv"
mouse_event_metrics.to_csv(mouse_event_out, index=False)

# -------------------------------------------------
# REM AMOUNT / QUALITY METRICS FROM EEG-ONLY MODEL
# -------------------------------------------------
manifest_df = load_all_manifests()

segment_rows = []
mouse_rem_rows = []

if len(manifest_df) > 0:
    # Keep only rows with EEG-only probability files.
    if "file_path_state_probabilities" in manifest_df.columns:
        prob_col = "file_path_state_probabilities"
    elif "file_path_state_probabilities_EEGonly" in manifest_df.columns:
        prob_col = "file_path_state_probabilities_EEGonly"
    else:
        prob_col = None

    if prob_col is not None:
        manifest_df = manifest_df[
            manifest_df[prob_col].map(lambda p: Path(p).exists() if isinstance(p, str) else False)
        ].copy()

    print("\nRows with EEG-only probability files:", len(manifest_df))

    # Prepare event lookup by segment.
    event_key_cols = ["recording_name", "mouse_id", "week", "segment_id"]

    for _, row in manifest_df.iterrows():
        probs, states, pred, conf = load_probabilities(row[prob_col])
        if probs is None:
            continue

        n_epochs = len(pred)

        P_awake = get_prob(probs, states, "Awake")
        P_nrem = get_prob(probs, states, "NREM")
        P_rem = get_prob(probs, states, "REM")

        if P_awake is None or P_rem is None:
            continue

        dist = transition_distance_sec(pred)

        eeg_rem = P_rem >= REM_PROB_STABLE
        stable_eeg_rem = (
            (P_rem >= REM_PROB_STABLE)
            & (P_awake <= WAKE_PROB_MAX_FOR_REM)
            & (dist >= TRANSITION_DISTANCE_SEC)
        )

        rem_bouts = get_bouts(eeg_rem)
        stable_rem_bouts = get_bouts(stable_eeg_rem)

        rem_bout_durations = np.array([(e - s + 1) * EPOCH_SEC / 60 for s, e in rem_bouts])
        stable_rem_bout_durations = np.array([(e - s + 1) * EPOCH_SEC / 60 for s, e in stable_rem_bouts])

        # Mark EMG burst epochs for this segment.
        seg_events = events[
            (events["recording_name"].astype(str) == str(row.get("recording_name", "")))
            & (events["mouse_id"].astype(int) == int(row.get("mouse_id")))
            & (events["week"].astype(int) == int(row.get("week")))
            & (events["segment_id"].astype(int) == int(row.get("segment_id")))
        ].copy()

        burst_epoch_mask = np.zeros(n_epochs, dtype=bool)

        for _, ev in seg_events.iterrows():
            s = int(np.clip(ev["start_epoch"], 0, n_epochs - 1))
            e = int(np.clip(ev["end_epoch"], 0, n_epochs - 1))
            burst_epoch_mask[s:e + 1] = True

        eeg_rem_epochs = int(eeg_rem.sum())
        stable_eeg_rem_epochs = int(stable_eeg_rem.sum())

        segment_rows.append({
            "recording_name": row.get("recording_name", ""),
            "group": row.get("group", ""),
            "week": row.get("week", ""),
            "mouse_id": row.get("mouse_id", ""),
            "segment_id": row.get("segment_id", ""),
            "n_epochs": n_epochs,
            "duration_min": n_epochs * EPOCH_SEC / 60,

            "EEGonly_REM_min": eeg_rem_epochs * EPOCH_SEC / 60,
            "stable_EEGonly_REM_min": stable_eeg_rem_epochs * EPOCH_SEC / 60,

            "n_EEGonly_REM_bouts": len(rem_bouts),
            "n_stable_EEGonly_REM_bouts": len(stable_rem_bouts),
            "mean_EEGonly_REM_bout_min": float(np.mean(rem_bout_durations)) if len(rem_bout_durations) else 0,
            "median_EEGonly_REM_bout_min": float(np.median(rem_bout_durations)) if len(rem_bout_durations) else 0,
            "short_EEGonly_REM_bouts_lt_30s": int(np.sum(rem_bout_durations < 0.5)) if len(rem_bout_durations) else 0,
            "REM_fragmentation_index_bouts_per_REM_hour": (
                len(rem_bouts) / (eeg_rem_epochs * EPOCH_SEC / 3600)
                if eeg_rem_epochs > 0 else np.nan
            ),

            "n_EMG_events": len(seg_events),
            "n_EMG_events_center_EEGonly_REM": int((seg_events["EEGonly_state_center"] == "REM").sum()) if len(seg_events) else 0,
            "n_candidate_EMG_suppressed_REM": int((seg_events["event_class"] == "candidate_EMG_suppressed_REM").sum()) if len(seg_events) else 0,
            "n_stable_EEG_REM_EMG_burst": int((seg_events["event_class"] == "stable_EEG_REM_EMG_burst").sum()) if len(seg_events) else 0,

            "pct_EEGonly_REM_epochs_with_EMG_burst": (
                np.mean(burst_epoch_mask & eeg_rem) / np.mean(eeg_rem) * 100
                if eeg_rem_epochs > 0 else np.nan
            ),
            "pct_stable_EEGonly_REM_epochs_with_EMG_burst": (
                np.mean(burst_epoch_mask & stable_eeg_rem) / np.mean(stable_eeg_rem) * 100
                if stable_eeg_rem_epochs > 0 else np.nan
            ),
        })

segment_metrics = pd.DataFrame(segment_rows)

if len(segment_metrics) > 0:
    segment_out = OUT_DIR / "segment_level_EEGonly_REM_quality_and_EMG_burden.csv"
    segment_metrics.to_csv(segment_out, index=False)

    mouse_rem = segment_metrics.groupby(["group", "week", "mouse_id"]).agg(
        total_recording_min=("duration_min", "sum"),
        total_EEGonly_REM_min=("EEGonly_REM_min", "sum"),
        total_stable_EEGonly_REM_min=("stable_EEGonly_REM_min", "sum"),
        total_EEGonly_REM_bouts=("n_EEGonly_REM_bouts", "sum"),
        total_stable_EEGonly_REM_bouts=("n_stable_EEGonly_REM_bouts", "sum"),
        total_short_EEGonly_REM_bouts_lt_30s=("short_EEGonly_REM_bouts_lt_30s", "sum"),
        total_EMG_events=("n_EMG_events", "sum"),
        total_EMG_events_center_EEGonly_REM=("n_EMG_events_center_EEGonly_REM", "sum"),
        total_candidate_EMG_suppressed_REM=("n_candidate_EMG_suppressed_REM", "sum"),
        total_stable_EEG_REM_EMG_burst=("n_stable_EEG_REM_EMG_burst", "sum"),
    ).reset_index()

    mouse_rem["EEGonly_REM_pct_recording"] = (
        mouse_rem["total_EEGonly_REM_min"] / mouse_rem["total_recording_min"] * 100
    )
    mouse_rem["stable_EEGonly_REM_pct_recording"] = (
        mouse_rem["total_stable_EEGonly_REM_min"] / mouse_rem["total_recording_min"] * 100
    )
    mouse_rem["EMG_events_per_recording_hour"] = (
        mouse_rem["total_EMG_events"] / (mouse_rem["total_recording_min"] / 60)
    )
    mouse_rem["EMG_events_per_EEGonly_REM_min"] = (
        mouse_rem["total_EMG_events_center_EEGonly_REM"] / mouse_rem["total_EEGonly_REM_min"]
    )
    mouse_rem["candidate_EMG_suppressed_REM_per_EEGonly_REM_min"] = (
        mouse_rem["total_candidate_EMG_suppressed_REM"] / mouse_rem["total_EEGonly_REM_min"]
    )
    mouse_rem["stable_EEG_REM_EMG_burst_per_EEGonly_REM_min"] = (
        mouse_rem["total_stable_EEG_REM_EMG_burst"] / mouse_rem["total_stable_EEGonly_REM_min"]
    )
    mouse_rem["REM_fragmentation_index_bouts_per_REM_hour"] = (
        mouse_rem["total_EEGonly_REM_bouts"] / (mouse_rem["total_EEGonly_REM_min"] / 60)
    )
    mouse_rem["short_REM_bout_fraction_pct"] = (
        mouse_rem["total_short_EEGonly_REM_bouts_lt_30s"] / mouse_rem["total_EEGonly_REM_bouts"] * 100
    )

    mouse_rem_out = OUT_DIR / "mouse_level_EEGonly_REM_quality_and_EMG_burden.csv"
    mouse_rem.to_csv(mouse_rem_out, index=False)

else:
    print("\nNo EEG-only REM-quality metrics calculated. Probability files may be missing.")
    segment_out = None
    mouse_rem = pd.DataFrame()
    mouse_rem_out = None

# -------------------------------------------------
# FINAL COMBINED MOUSE TABLE
# -------------------------------------------------
if len(mouse_rem) > 0:
    combined = mouse_rem.merge(
        mouse_event_metrics,
        on=["group", "week", "mouse_id"],
        how="outer",
    )
else:
    combined = mouse_event_metrics.copy()

combined_out = OUT_DIR / "mouse_level_supervised_RBD_metrics_combined.csv"
combined.to_csv(combined_out, index=False)

# -------------------------------------------------
# GROUP/WEEK SUMMARY
# -------------------------------------------------
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["week", "mouse_id"]]

group_summary = combined.groupby(["group", "week"])[numeric_cols].agg(["mean", "std", "count"])
group_summary.columns = ["_".join(col).strip() for col in group_summary.columns.values]
group_summary = group_summary.reset_index()

group_summary_out = OUT_DIR / "group_week_supervised_RBD_metrics_summary.csv"
group_summary.to_csv(group_summary_out, index=False)

# Event class group/week counts
event_class_group = pd.crosstab(
    [events["group"], events["week"]],
    events["event_class"]
)
event_class_group.to_csv(OUT_DIR / "group_week_event_class_counts.csv")

event_class_group_pct = pd.crosstab(
    [events["group"], events["week"]],
    events["event_class"],
    normalize="index"
) * 100
event_class_group_pct.to_csv(OUT_DIR / "group_week_event_class_percentages.csv")

print("\nWrote outputs to:")
print(OUT_DIR)
print()
print("Main files:")
print(" ", combined_out)
print(" ", group_summary_out)
print(" ", mouse_event_out)
if len(segment_metrics) > 0:
    print(" ", segment_out)
    print(" ", mouse_rem_out)

print("\nEvent class counts:")
print(events["event_class"].value_counts().to_string())

print("\nEvent class percentages by group/week:")
print(event_class_group_pct.round(2).to_string())

print("\nCombined mouse-level metrics preview:")
print(combined.head(20).to_string(index=False))

print("\nGroup/week summary preview:")
print(group_summary.head(20).to_string(index=False))
