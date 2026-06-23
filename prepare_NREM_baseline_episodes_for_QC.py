from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")

IN = BASE / "EMG_burst_detection_NREM_baseline/EMG_episodes_NREMbaseline_on4_off2_episodegap10s.csv"

OUT_DIR = BASE / "EMG_burst_detection_NREM_baseline/qc_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "EMG_episodes_NREMbaseline_qc_ready.csv"
CRITERIA = OUT_DIR / "NREMbaseline_episode_category_criteria.csv"

df = pd.read_csv(IN)

df = df.copy()
df.insert(0, "qc_event_id", np.arange(len(df)))

df["primary_category"] = df["event_class"]

df["P_REM_EEGonly"] = df["mean_EEGonly_P_REM"]
df["P_Wake_EEGonly"] = df["mean_EEGonly_P_Awake"]
df["P_NREM_EEGonly"] = df["mean_EEGonly_P_NREM"]

df["P_REM_FULL"] = df["mean_full_P_REM"]
df["P_Wake_FULL"] = df["mean_full_P_Awake"]
df["P_NREM_FULL"] = df["mean_full_P_NREM"]

df["delta_REM"] = df["mean_delta_REM_EEGonly_minus_full"]
df["distance_to_transition_sec"] = df["min_EEGonly_distance_to_transition_sec"]

df["duration_sec_for_category"] = df["duration_sec"]

# For the app, map the new baseline z to max_EMG_z.
df["max_EMG_z"] = df["max_EMG_baseline_z"]

# Useful ranking score for RBD-like QC.
df["rbd_priority_score"] = (
    2.0 * df["P_REM_EEGonly"].fillna(0)
    + 2.0 * df["delta_REM"].fillna(0)
    + 0.5 * df["max_EMG_z"].fillna(0)
    - 1.0 * df["P_Wake_EEGonly"].fillna(0)
)

# Mark REM-relevant subset.
df["REM_relevant_episode"] = df["primary_category"].isin([
    "stable_REM_EMG_burst",
    "EMG_suppressed_REM",
    "mixed_REM_Wake_transition",
])

df["phasic_episode_0p1_to_5s"] = df["duration_sec_for_category"].between(0.1, 5.0)
df["long_episode_gt_5s"] = df["duration_sec_for_category"] > 5.0

df.to_csv(OUT, index=False)

criteria = pd.DataFrame([
    {
        "category": "stable_REM_EMG_burst",
        "criteria": "EEG-only P(REM) high, EEG-only P(Wake) low, far from transition",
        "interpretation": "REM-like brain state with EMG episode; strongest RBD-like category if QC confirms signal.",
    },
    {
        "category": "EMG_suppressed_REM",
        "criteria": "EEG-only P(REM) high, full-model P(REM) lower, positive ΔREM",
        "interpretation": "EEG looks REM-like, but adding EMG suppresses REM probability.",
    },
    {
        "category": "mixed_REM_Wake_transition",
        "criteria": "Intermediate REM/Wake probabilities or close to transition",
        "interpretation": "Motor activity near REM/Wake instability or transition.",
    },
    {
        "category": "wake_like_movement",
        "criteria": "EEG-only P(Wake) high and P(REM) low",
        "interpretation": "Likely ordinary wake movement.",
    },
    {
        "category": "NREM_like_EMG",
        "criteria": "EEG-only P(NREM) high",
        "interpretation": "EMG episode during NREM-like state; useful control but not primary RBD category.",
    },
    {
        "category": "other_uncertain",
        "criteria": "Does not clearly match above categories",
        "interpretation": "Needs QC or refined thresholding.",
    },
])
criteria.to_csv(CRITERIA, index=False)

print("Wrote:")
print(OUT)
print(CRITERIA)

print("\nQC-ready category counts:")
print(df["primary_category"].value_counts().to_string())

print("\nREM-relevant counts by group/week:")
rem = df[df["REM_relevant_episode"]]
print(pd.crosstab([rem["group"], rem["week"]], rem["primary_category"]).to_string())
