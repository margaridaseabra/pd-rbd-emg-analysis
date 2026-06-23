from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime


def stable_event_key(row):
    event_type = row.get("event_type", "episode")
    fields = [
        row.get("recording_name", ""),
        row.get("group", ""),
        int(row.get("week", -1)) if pd.notna(row.get("week", np.nan)) else -1,
        int(row.get("mouse_id", -1)) if pd.notna(row.get("mouse_id", np.nan)) else -1,
        int(row.get("segment_id", -1)) if pd.notna(row.get("segment_id", np.nan)) else -1,
        event_type,
        f"{float(row.get('start_sec', 0)):.1f}",
        f"{float(row.get('end_sec', 0)):.1f}",
    ]
    return hashlib.sha1("|".join(map(str, fields)).encode()).hexdigest()[:20]


def ensure_keys(df):
    df = df.copy()
    if "stable_event_key" not in df.columns:
        df["stable_event_key"] = df.apply(stable_event_key, axis=1)
    else:
        missing = df["stable_event_key"].isna() | (df["stable_event_key"].astype(str) == "")
        df.loc[missing, "stable_event_key"] = df.loc[missing].apply(stable_event_key, axis=1)
    return df


def interval_overlap(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0


def attach_qc_metadata(qc, old_events):
    """
    Ensure old QC rows have timing/stable key metadata by merging with old event table.
    """
    qc = qc.copy()
    old_events = ensure_keys(old_events)

    cols = [
        "qc_event_id",
        "stable_event_key",
        "recording_name",
        "group",
        "week",
        "mouse_id",
        "segment_id",
        "event_type",
        "start_sec",
        "end_sec",
        "primary_category",
    ]
    cols = [c for c in cols if c in old_events.columns]

    old_map = old_events[cols].drop_duplicates("qc_event_id")

    # Avoid duplicate columns if QC already has some metadata.
    drop_cols = [c for c in old_map.columns if c != "qc_event_id" and c in qc.columns]
    qc = qc.drop(columns=drop_cols, errors="ignore")

    qc = qc.merge(old_map, on="qc_event_id", how="left")

    if "stable_event_key" not in qc.columns:
        qc["stable_event_key"] = qc.apply(stable_event_key, axis=1)

    return qc


def fuzzy_match_one(qc_row, new_events, tolerance_sec=2.0, min_overlap=0.30):
    same = new_events[
        (new_events["recording_name"].astype(str) == str(qc_row.get("recording_name", "")))
        & (new_events["group"].astype(str) == str(qc_row.get("group", "")))
        & (new_events["week"].astype(int) == int(qc_row.get("week", -999)))
        & (new_events["mouse_id"].astype(int) == int(qc_row.get("mouse_id", -999)))
        & (new_events["segment_id"].astype(int) == int(qc_row.get("segment_id", -999)))
    ].copy()

    if len(same) == 0:
        return None, "no_same_segment"

    if "event_type" in same.columns and pd.notna(qc_row.get("event_type", np.nan)):
        same = same[same["event_type"].astype(str) == str(qc_row.get("event_type"))].copy()

    if len(same) == 0:
        return None, "no_same_event_type"

    q0 = float(qc_row.get("start_sec", np.nan))
    q1 = float(qc_row.get("end_sec", np.nan))
    qc_center = (q0 + q1) / 2

    same["center_sec"] = (same["start_sec"].astype(float) + same["end_sec"].astype(float)) / 2
    same["center_diff"] = np.abs(same["center_sec"] - qc_center)
    same["overlap_score"] = same.apply(
        lambda r: interval_overlap(q0, q1, float(r["start_sec"]), float(r["end_sec"])),
        axis=1,
    )

    candidates = same[
        (same["center_diff"] <= tolerance_sec)
        | (same["overlap_score"] >= min_overlap)
    ].copy()

    if len(candidates) == 0:
        return None, "no_temporal_match"

    candidates = candidates.sort_values(["overlap_score", "center_diff"], ascending=[False, True])
    best = candidates.iloc[0]

    return best, "fuzzy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-events", required=True)
    parser.add_argument("--old-qc", required=True)
    parser.add_argument("--new-events", required=True)
    parser.add_argument("--out-qc", required=True)
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    args = parser.parse_args()

    old_events = pd.read_csv(args.old_events)
    old_qc = pd.read_csv(args.old_qc)
    new_events = pd.read_csv(args.new_events)

    old_events = ensure_keys(old_events)
    new_events = ensure_keys(new_events)
    old_qc = attach_qc_metadata(old_qc, old_events)

    transferred = []
    unmatched = []

    new_by_key = new_events.drop_duplicates("stable_event_key").set_index("stable_event_key")

    for _, q in old_qc.iterrows():
        q = q.copy()
        key = q.get("stable_event_key", "")

        match_method = None
        new_row = None

        if key in new_by_key.index:
            new_row = new_by_key.loc[key]
            match_method = "stable_event_key"
        else:
            new_row, match_method = fuzzy_match_one(
                q,
                new_events,
                tolerance_sec=args.tolerance_sec,
            )

        if new_row is None:
            out = q.to_dict()
            out["match_method"] = match_method
            unmatched.append(out)
            continue

        new_q = q.to_dict()
        new_q["old_qc_event_id"] = q.get("qc_event_id")
        new_q["qc_event_id"] = int(new_row["qc_event_id"])
        new_q["stable_event_key"] = new_row["stable_event_key"]
        new_q["match_method"] = match_method

        # Refresh event metadata from new table.
        for col in [
            "recording_name",
            "group",
            "week",
            "mouse_id",
            "segment_id",
            "event_type",
            "start_sec",
            "end_sec",
            "primary_category",
            "manual_state_center",
            "EEGonly_state_center",
            "full_state_center",
        ]:
            if col in new_row.index:
                new_q[col] = new_row[col]

        transferred.append(new_q)

    out_qc = pd.DataFrame(transferred)
    unmatched_df = pd.DataFrame(unmatched)

    out_path = Path(args.out_qc)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_path.exists():
        backup = out_path.with_name(out_path.stem + f"_BACKUP_before_transfer_{stamp}" + out_path.suffix)
        backup.write_bytes(out_path.read_bytes())
        print("Backed up old out-qc:", backup)

    out_qc.to_csv(out_path, index=False)

    unmatched_path = out_path.with_name(out_path.stem + "_UNMATCHED_after_transfer.csv")
    unmatched_df.to_csv(unmatched_path, index=False)

    # Also write new event table with stable keys if missing.
    new_events.to_csv(args.new_events, index=False)

    print("\nTransfer complete.")
    print("Old QC rows:", len(old_qc))
    print("Transferred:", len(out_qc))
    print("Unmatched:", len(unmatched_df))
    print("Output QC:", out_path)
    print("Unmatched file:", unmatched_path)

    if len(out_qc):
        print("\nMatch methods:")
        print(out_qc["match_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
