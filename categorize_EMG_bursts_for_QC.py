from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")

EVENT_TABLE = BASE / "EMG_burst_detection_EEGonly_scored/filtered_nontriplewake/EMG_burst_events_NONTRIPLEWAKE_highconf_filter.csv"

FULL_APP = Path.home() / "Desktop/local_sleep_manifests/final_WT_reference_manifests/WT_PD_week2_week21_FULL_finalWTref_application.csv"
EEG_APP = Path.home() / "Desktop/local_sleep_manifests/final_WT_reference_manifests/WT_PD_week2_week21_EEGonly_finalWTref_application.csv"

OUT_DIR = BASE / "EMG_burst_detection_EEGonly_scored/categorized_qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENTS = OUT_DIR / "EMG_burst_events_categorized_for_QC.csv"
OUT_CRITERIA = OUT_DIR / "EMG_burst_category_criteria.csv"
OUT_SUMMARY = OUT_DIR / "EMG_burst_category_summary.csv"
OUT_MOUSE = OUT_DIR / "EMG_burst_category_mouse_summary.csv"

# ---------------- THRESHOLDS ----------------
P_REM_HIGH = 0.70
P_REM_INTERMEDIATE_LOW = 0.30
P_REM_INTERMEDIATE_HIGH = 0.70

P_WAKE_LOW = 0.30
P_WAKE_HIGH = 0.70
P_WAKE_INTERMEDIATE_LOW = 0.20
P_WAKE_INTERMEDIATE_HIGH = 0.70

P_NREM_HIGH = 0.70

DELTA_REM_LARGE = 0.25
FULL_REM_SUPPRESSED_MAX = 0.60

AMBIGUITY_HIGH = 0.25
TRANSITION_NEAR_SEC = 30
TRANSITION_FAR_SEC = 30

PHASIC_MIN_SEC = 0.10
PHASIC_MAX_SEC = 5.00


def first_existing_col(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def normalize_state(x):
    x = str(x).strip()
    mapping = {
        "Wake": "Awake",
        "WK": "Awake",
        "W": "Awake",
        "wake": "Awake",
        "AWAKE": "Awake",
        "Awake": "Awake",
        "SWS": "NREM",
        "NREM": "NREM",
        "Nrem": "NREM",
        "PS": "REM",
        "REM": "REM",
        "Rem": "REM",
        "TR": "Undefined",
        "ND": "Undefined",
        "Undefined": "Undefined",
        "nan": "Undefined",
        "": "Undefined",
    }
    return mapping.get(x, x)


def add_manifest_paths(events):
    keys = ["recording_name", "group", "week", "mouse_id", "segment_id"]

    full = pd.read_csv(FULL_APP)
    eeg = pd.read_csv(EEG_APP)

    full_cols = keys + [
        "file_path_state_probabilities",
        "file_path_automated_state_annotation",
        "file_path_manual_state_annotation",
        "file_path_raw_signals",
        "file_path_preprocessed_signals",
    ]
    full_cols = [c for c in full_cols if c in full.columns]

    eeg_cols = keys + [
        "file_path_state_probabilities",
        "file_path_automated_state_annotation",
        "file_path_preprocessed_signals",
    ]
    eeg_cols = [c for c in eeg_cols if c in eeg.columns]

    full_small = full[full_cols].copy()
    eeg_small = eeg[eeg_cols].copy()

    rename_full = {
        "file_path_state_probabilities": "file_path_state_probabilities_FULL",
        "file_path_automated_state_annotation": "file_path_automated_state_annotation_FULL",
        "file_path_preprocessed_signals": "file_path_preprocessed_signals_FULL",
    }

    rename_eeg = {
        "file_path_state_probabilities": "file_path_state_probabilities_EEGonly",
        "file_path_automated_state_annotation": "file_path_automated_state_annotation_EEGonly",
        "file_path_preprocessed_signals": "file_path_preprocessed_signals_EEGonly",
    }

    full_small = full_small.rename(columns=rename_full)
    eeg_small = eeg_small.rename(columns=rename_eeg)

    out = events.merge(full_small, on=keys, how="left", suffixes=("", "_from_full_manifest"))
    out = out.merge(eeg_small, on=keys, how="left", suffixes=("", "_from_eeg_manifest"))

    return out


def categorize(df):
    df = df.copy()

    if "qc_event_id" not in df.columns:
        df.insert(0, "qc_event_id", np.arange(len(df)))

    df["P_REM_EEGonly"] = pd.to_numeric(first_existing_col(df, ["mean_EEGonly_P_REM"]), errors="coerce")
    df["P_Wake_EEGonly"] = pd.to_numeric(first_existing_col(df, ["mean_EEGonly_P_Awake"]), errors="coerce")
    df["P_NREM_EEGonly"] = pd.to_numeric(first_existing_col(df, ["mean_EEGonly_P_NREM"]), errors="coerce")

    df["P_REM_FULL"] = pd.to_numeric(first_existing_col(df, ["mean_full_P_REM"]), errors="coerce")
    df["P_Wake_FULL"] = pd.to_numeric(first_existing_col(df, ["mean_full_P_Awake"]), errors="coerce")

    df["delta_REM"] = pd.to_numeric(
        first_existing_col(df, ["mean_delta_REM_EEGonly_minus_full"]),
        errors="coerce",
    )

    # If delta column is missing, compute it.
    miss_delta = df["delta_REM"].isna()
    df.loc[miss_delta, "delta_REM"] = (
        df.loc[miss_delta, "P_REM_EEGonly"] - df.loc[miss_delta, "P_REM_FULL"]
    )

    df["EEGonly_ambiguity"] = pd.to_numeric(
        first_existing_col(df, ["mean_EEGonly_ambiguity"]),
        errors="coerce",
    )

    df["distance_to_transition_sec"] = pd.to_numeric(
        first_existing_col(df, ["min_EEGonly_distance_to_transition_sec"]),
        errors="coerce",
    )

    df["duration_sec_for_category"] = pd.to_numeric(
        first_existing_col(df, ["duration_sec", "event_duration_sec", "duration_s"]),
        errors="coerce",
    )

    for col in ["manual_state_center", "EEGonly_state_center", "full_state_center"]:
        if col in df.columns:
            df[col + "_norm"] = df[col].map(normalize_state)
        else:
            df[col + "_norm"] = "Undefined"

    df["flag_phasic_duration_0p1_to_5s"] = (
        (df["duration_sec_for_category"] >= PHASIC_MIN_SEC)
        & (df["duration_sec_for_category"] <= PHASIC_MAX_SEC)
    )

    df["flag_stable_REM"] = (
        (df["P_REM_EEGonly"] >= P_REM_HIGH)
        & (df["P_Wake_EEGonly"] <= P_WAKE_LOW)
        & (df["distance_to_transition_sec"] >= TRANSITION_FAR_SEC)
    )

    df["flag_EMG_suppressed_REM"] = (
        (df["P_REM_EEGonly"] >= P_REM_HIGH)
        & (df["delta_REM"] >= DELTA_REM_LARGE)
        & (df["P_REM_FULL"] <= FULL_REM_SUPPRESSED_MAX)
    )

    df["flag_mixed_REM_Wake_transition"] = (
        (
            (df["P_REM_EEGonly"] >= P_REM_INTERMEDIATE_LOW)
            & (df["P_REM_EEGonly"] < P_REM_INTERMEDIATE_HIGH)
            & (df["P_Wake_EEGonly"] >= P_WAKE_INTERMEDIATE_LOW)
            & (df["P_Wake_EEGonly"] <= P_WAKE_INTERMEDIATE_HIGH)
        )
        | (df["EEGonly_ambiguity"] >= AMBIGUITY_HIGH)
        | (
            (df["distance_to_transition_sec"] < TRANSITION_NEAR_SEC)
            & (df["P_REM_EEGonly"] >= P_REM_INTERMEDIATE_LOW)
        )
    )

    df["flag_wake_like"] = (
        (df["P_Wake_EEGonly"] >= P_WAKE_HIGH)
        & (df["P_REM_EEGonly"] <= P_WAKE_LOW)
    )

    df["flag_NREM_like"] = (
        (df["P_NREM_EEGonly"] >= P_NREM_HIGH)
        & (df["P_REM_EEGonly"] < P_REM_INTERMEDIATE_LOW)
        & (df["P_Wake_EEGonly"] < P_WAKE_HIGH)
    )

    # Primary category priority:
    # EMG-suppressed REM first because it captures the model-disagreement pattern.
    primary = []

    for _, r in df.iterrows():
        if bool(r["flag_EMG_suppressed_REM"]):
            primary.append("EMG_suppressed_REM")
        elif bool(r["flag_stable_REM"]):
            primary.append("stable_REM_EMG_burst")
        elif bool(r["flag_mixed_REM_Wake_transition"]):
            primary.append("mixed_REM_Wake_transition")
        elif bool(r["flag_wake_like"]):
            primary.append("wake_like_movement")
        elif bool(r["flag_NREM_like"]):
            primary.append("NREM_like_EMG")
        else:
            primary.append("other_uncertain")

    df["primary_category"] = primary

    # More detailed labels for interpretation.
    df["duration_category"] = np.where(
        df["duration_sec_for_category"] <= PHASIC_MAX_SEC,
        "phasic_0p1_to_5s",
        "long_sustained_gt_5s",
    )

    df["rbd_priority_score"] = (
        2.0 * df["P_REM_EEGonly"].fillna(0)
        + 2.0 * df["delta_REM"].fillna(0)
        + 0.5 * df["max_EMG_z"].fillna(0)
        - 1.0 * df["P_Wake_EEGonly"].fillna(0)
    )

    return df


if not EVENT_TABLE.exists():
    raise FileNotFoundError(f"Could not find event table: {EVENT_TABLE}")

events = pd.read_csv(EVENT_TABLE)
events = add_manifest_paths(events)
events = categorize(events)

events.to_csv(OUT_EVENTS, index=False)

criteria = pd.DataFrame([
    {
        "category": "stable_REM_EMG_burst",
        "criteria": f"EEG-only P(REM) >= {P_REM_HIGH}; EEG-only P(Wake) <= {P_WAKE_LOW}; distance to EEG-only transition >= {TRANSITION_FAR_SEC} s",
        "interpretation": "Strongest REM-like brain-state with EMG activity; candidate RBD-like event if QC confirms EMG burst.",
    },
    {
        "category": "EMG_suppressed_REM",
        "criteria": f"EEG-only P(REM) >= {P_REM_HIGH}; delta REM >= {DELTA_REM_LARGE}; full P(REM) <= {FULL_REM_SUPPRESSED_MAX}",
        "interpretation": "EEG-only model sees REM, but adding EMG reduces REM probability; directly tests EMG interference with REM classification.",
    },
    {
        "category": "mixed_REM_Wake_transition",
        "criteria": f"Intermediate EEG-only P(REM) {P_REM_INTERMEDIATE_LOW}-{P_REM_INTERMEDIATE_HIGH}; intermediate P(Wake), ambiguity >= {AMBIGUITY_HIGH}, or transition distance < {TRANSITION_NEAR_SEC} s",
        "interpretation": "Motor activity during uncertain REM/Wake boundary or transition.",
    },
    {
        "category": "wake_like_movement",
        "criteria": f"EEG-only P(Wake) >= {P_WAKE_HIGH}; EEG-only P(REM) <= {P_WAKE_LOW}",
        "interpretation": "Likely ordinary Wake movement rather than RBD-like event.",
    },
    {
        "category": "NREM_like_EMG",
        "criteria": f"EEG-only P(NREM) >= {P_NREM_HIGH}; low REM and not high Wake",
        "interpretation": "EMG burst during NREM-like state.",
    },
    {
        "category": "other_uncertain",
        "criteria": "Does not satisfy any primary category criteria",
        "interpretation": "Needs QC or refined thresholding.",
    },
])
criteria.to_csv(OUT_CRITERIA, index=False)

summary = events.groupby(["group", "week", "primary_category"]).agg(
    n_events=("qc_event_id", "count"),
    mean_P_REM_EEGonly=("P_REM_EEGonly", "mean"),
    mean_P_Wake_EEGonly=("P_Wake_EEGonly", "mean"),
    mean_P_REM_FULL=("P_REM_FULL", "mean"),
    mean_delta_REM=("delta_REM", "mean"),
    mean_max_EMG_z=("max_EMG_z", "mean"),
    mean_duration_sec=("duration_sec_for_category", "mean"),
).reset_index()

summary.to_csv(OUT_SUMMARY, index=False)

mouse_summary = events.groupby(["group", "week", "mouse_id", "primary_category"]).agg(
    n_events=("qc_event_id", "count"),
    mean_P_REM_EEGonly=("P_REM_EEGonly", "mean"),
    mean_delta_REM=("delta_REM", "mean"),
    mean_max_EMG_z=("max_EMG_z", "mean"),
).reset_index()

mouse_summary.to_csv(OUT_MOUSE, index=False)

print("\nWrote:")
print(OUT_EVENTS)
print(OUT_CRITERIA)
print(OUT_SUMMARY)
print(OUT_MOUSE)

print("\nCategory criteria:")
print(criteria.to_string(index=False))

print("\nCategory counts:")
print(events["primary_category"].value_counts().to_string())

print("\nGroup/week/category summary:")
print(pd.crosstab([events["group"], events["week"]], events["primary_category"]).to_string())
