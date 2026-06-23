from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# ---------------- USER SETTINGS ----------------
PREPARED_ROOT = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data"
)

WT_MICE = {1, 7, 8, 10, 11}

MANIFESTS_DIR = PREPARED_ROOT / "manifests"
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------------


def infer_mouse_id(name: str) -> int | None:
    patterns = [
        r"(?i)(?:^|_)Mouse(\d+)(?:_|$)",
        r"(?i)(?:^|_)M(\d+)(?:_|$)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def infer_week(name: str) -> int | None:
    patterns = [
        r"(?i)wk(\d+)",
        r"(?i)(\d+)wk",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def infer_group(mouse_id: int | None) -> str:
    if mouse_id is None:
        return "unknown"
    return "WT" if mouse_id in WT_MICE else "PD"


def segment_id_from_path(raw_path: str) -> int | None:
    m = re.search(r"segment_(\d+)", raw_path)
    if m:
        return int(m.group(1))
    return None


def summarize_state_table(segment_dir: Path) -> dict:
    state_table_path = segment_dir / "state_table_5s.csv"
    if not state_table_path.exists():
        return {
            "n_epochs": pd.NA,
            "n_wk": pd.NA,
            "n_sws": pd.NA,
            "n_ps": pd.NA,
            "n_nd": pd.NA,
            "n_tr": pd.NA,
            "n_artef": pd.NA,
            "n_scored": pd.NA,
            "pct_scored": pd.NA,
            "pct_undefined_like": pd.NA,
            "has_rem": pd.NA,
        }

    df = pd.read_csv(state_table_path)
    if "label_raw" not in df.columns or len(df) == 0:
        return {
            "n_epochs": 0,
            "n_wk": 0,
            "n_sws": 0,
            "n_ps": 0,
            "n_nd": 0,
            "n_tr": 0,
            "n_artef": 0,
            "n_scored": 0,
            "pct_scored": 0.0,
            "pct_undefined_like": 1.0,
            "has_rem": False,
        }

    counts = df["label_raw"].fillna("MISSING").value_counts()

    n_epochs = int(len(df))
    n_wk = int(counts.get("WK", 0))
    n_sws = int(counts.get("SWS", 0))
    n_ps = int(counts.get("PS", 0))
    n_nd = int(counts.get("ND", 0))
    n_tr = int(counts.get("TR", 0))
    n_artef = int(counts.get("Artef", 0))

    n_scored = n_wk + n_sws + n_ps
    pct_scored = n_scored / n_epochs if n_epochs else 0.0
    pct_undefined_like = (n_nd + n_tr + n_artef) / n_epochs if n_epochs else 1.0

    return {
        "n_epochs": n_epochs,
        "n_wk": n_wk,
        "n_sws": n_sws,
        "n_ps": n_ps,
        "n_nd": n_nd,
        "n_tr": n_tr,
        "n_artef": n_artef,
        "n_scored": n_scored,
        "pct_scored": pct_scored,
        "pct_undefined_like": pct_undefined_like,
        "has_rem": n_ps > 0,
    }


def recommended_use(group: str, week: int | None, n_scored: int | None, has_manual_labels: bool) -> str:
    if not has_manual_labels:
        return "inference_only_unlabeled"
    if n_scored is not None and n_scored == 0:
        return "exclude_all_undefined"
    if group == "WT" and week == 2:
        return "wt_train_candidate"
    if group == "PD" and week == 2:
        return "pd_wk2_inference_eval_candidate"
    if group == "PD" and week == 21:
        return "pd_wk21_review_only"
    return "review_manually"


def label_quality(group: str, week: int | None) -> str:
    if group == "WT" and week == 2:
        return "trusted"
    if group == "PD" and week == 2:
        return "check"
    if group == "PD" and week == 21:
        return "uncertain"
    return "unknown"


rows = []

for rec_dir in sorted(PREPARED_ROOT.iterdir()):
    if not rec_dir.is_dir():
        continue
    if rec_dir.name in {"manifests", "models", "results"}:
        continue

    som_manifest = rec_dir / "somnotate_manifest.csv"
    rec_meta = rec_dir / "recording_meta.json"

    if not som_manifest.exists():
        continue

    meta = {}
    if rec_meta.exists():
        meta = json.loads(rec_meta.read_text())

    rec_name = rec_dir.name
    mouse_id = infer_mouse_id(rec_name)
    week = infer_week(rec_name)
    group = infer_group(mouse_id)

    df = pd.read_csv(som_manifest)

    for _, r in df.iterrows():
        seg_id = segment_id_from_path(str(r["file_path_raw_signals"]))
        seg_dir = rec_dir / f"segment_{seg_id:02d}" if seg_id is not None else rec_dir
        stats = summarize_state_table(seg_dir)

        manual_path = Path(str(r["file_path_manual_state_annotation"]))
        prep_path = Path(str(r["file_path_preprocessed_signals"]))
        auto_path = Path(str(r["file_path_automated_state_annotation"]))
        prob_path = Path(str(r["file_path_state_probabilities"]))

        has_manual = manual_path.exists()
        has_preprocessed = prep_path.exists()
        has_auto = auto_path.exists()
        has_probs = prob_path.exists()

        rec = {
            "recording_name": rec_name,
            "recording_dir": str(rec_dir),
            "mouse_id": mouse_id,
            "group": group,
            "week": week,
            "segment_id": seg_id,
            "sampling_frequency_in_hz": r.get("sampling_frequency_in_hz", meta.get("sampling_rate")),
            "frontal_eeg_signal_label": r.get("frontal_eeg_signal_label", ""),
            "emg_signal_label": r.get("emg_signal_label", ""),
            "file_path_raw_signals": str(r["file_path_raw_signals"]),
            "file_path_preprocessed_signals": str(r["file_path_preprocessed_signals"]),
            "file_path_manual_state_annotation": str(r["file_path_manual_state_annotation"]),
            "file_path_automated_state_annotation": str(r["file_path_automated_state_annotation"]),
            "file_path_review_intervals": str(r["file_path_review_intervals"]),
            "file_path_state_probabilities": str(r["file_path_state_probabilities"]),
            "has_manual_labels": has_manual,
            "has_preprocessed": has_preprocessed,
            "has_auto_annotation": has_auto,
            "has_state_probabilities": has_probs,
            "label_quality": label_quality(group, week),
            **stats,
        }

        rec["recommended_use"] = recommended_use(
            group=group,
            week=week,
            n_scored=rec["n_scored"] if pd.notna(rec["n_scored"]) else None,
            has_manual_labels=has_manual,
        )

        rec["usable_for_training"] = bool(
            rec["recommended_use"] == "wt_train_candidate"
            and rec["has_manual_labels"]
            and rec["has_preprocessed"]
            and pd.notna(rec["n_scored"])
            and rec["n_scored"] > 0
        )

        rows.append(rec)

all_df = pd.DataFrame(rows).sort_values(
    ["group", "week", "mouse_id", "recording_name", "segment_id"]
).reset_index(drop=True)

all_path = MANIFESTS_DIR / "all_segments.csv"
all_df.to_csv(all_path, index=False)

wt_week2_all = all_df[
    (all_df["group"] == "WT") &
    (all_df["week"] == 2)
].copy()
wt_week2_all.to_csv(MANIFESTS_DIR / "wt_week2_all.csv", index=False)

wt_week2_train = all_df[
    (all_df["group"] == "WT") &
    (all_df["week"] == 2) &
    (all_df["usable_for_training"] == True)
].copy()
wt_week2_train.to_csv(MANIFESTS_DIR / "wt_week2_train.csv", index=False)

pd_week2_inference = all_df[
    (all_df["group"] == "PD") &
    (all_df["week"] == 2)
].copy()
pd_week2_inference.to_csv(MANIFESTS_DIR / "pd_week2_inference.csv", index=False)

pd_week21_review_only = all_df[
    (all_df["group"] == "PD") &
    (all_df["week"] == 21)
].copy()
pd_week21_review_only.to_csv(MANIFESTS_DIR / "pd_week21_review_only.csv", index=False)

print(f"Wrote: {all_path}")
print(f"Wrote: {MANIFESTS_DIR / 'wt_week2_all.csv'}")
print(f"Wrote: {MANIFESTS_DIR / 'wt_week2_train.csv'}")
print(f"Wrote: {MANIFESTS_DIR / 'pd_week2_inference.csv'}")
print(f"Wrote: {MANIFESTS_DIR / 'pd_week21_review_only.csv'}")