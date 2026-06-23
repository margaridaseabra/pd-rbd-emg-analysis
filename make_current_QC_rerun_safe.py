from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime

BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")
QC_READY = BASE / "EMG_burst_detection_NREM_baseline/qc_ready"

EVENTS = QC_READY / "EMG_episodes_NREMbaseline_qc_ready.csv"
QC = QC_READY / "interactive_QC_annotations.csv"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = QC_READY / f"backup_stable_event_key_{STAMP}"
BACKUP.mkdir(parents=True, exist_ok=True)


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


# Backups
for p in [EVENTS, QC]:
    if p.exists():
        out = BACKUP / p.name
        out.write_bytes(p.read_bytes())
        print("Backed up:", out)

events = pd.read_csv(EVENTS)

if "stable_event_key" not in events.columns:
    events["stable_event_key"] = events.apply(stable_event_key, axis=1)
else:
    missing = events["stable_event_key"].isna() | (events["stable_event_key"].astype(str) == "")
    events.loc[missing, "stable_event_key"] = events.loc[missing].apply(stable_event_key, axis=1)

events.to_csv(EVENTS, index=False)

print("\nEvent table updated:")
print(EVENTS)
print("Rows:", len(events))
print("Unique stable_event_key:", events["stable_event_key"].nunique())

if not QC.exists():
    print("\nNo QC annotations file found yet.")
    raise SystemExit

qc = pd.read_csv(QC)

# Attach stable_event_key to old QC annotations using current qc_event_id mapping.
map_cols = [
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
map_cols = [c for c in map_cols if c in events.columns]

event_map = events[map_cols].drop_duplicates("qc_event_id")

if "stable_event_key" in qc.columns:
    qc = qc.drop(columns=["stable_event_key"])

qc = qc.merge(event_map, on="qc_event_id", how="left", suffixes=("", "_event"))

# Prefer event-table metadata where available, but keep existing QC columns.
for c in ["recording_name", "group", "week", "mouse_id", "segment_id", "primary_category"]:
    event_c = f"{c}_event"
    if event_c in qc.columns:
        if c in qc.columns:
            qc[c] = qc[c].where(qc[c].notna() & (qc[c].astype(str) != ""), qc[event_c])
            qc = qc.drop(columns=[event_c])
        else:
            qc = qc.rename(columns={event_c: c})

qc.to_csv(QC, index=False)

print("\nQC annotations updated:")
print(QC)
print("Rows:", len(qc))
print("Rows with stable_event_key:", qc["stable_event_key"].notna().sum())
print("\nBackup folder:")
print(BACKUP)
