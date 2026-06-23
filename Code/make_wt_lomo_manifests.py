from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv")
OUT_DIR = MANIFEST.parent / "wt_lomo_validation"
OUT_DIR.mkdir(exist_ok=True)

EPOCH_SEC = 5

df = pd.read_csv(MANIFEST)

def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

def read_stage_duration(path):
    path = Path(path)
    lines = [l.rstrip("\n") for l in path.read_text().splitlines() if l.strip()]

    if not lines or not lines[0].startswith("*Duration"):
        raise ValueError(f"{path} does not look like stage-duration format")

    header2 = lines[1]
    entries = []

    for line in lines[2:]:
        parts = line.replace(",", "\t").split()
        if len(parts) < 2:
            continue
        label = " ".join(parts[:-1])
        end_sec = float(parts[-1])
        entries.append([label, end_sec])

    return header2, entries

def write_stage_duration(path, duration_sec, header2, entries):
    path = Path(path)
    with path.open("w") as f:
        f.write(f"*Duration_sec\t{int(duration_sec)}\n")
        f.write(header2 + "\n")
        for label, end_sec in entries:
            f.write(f"{label}\t{int(end_sec)}\n")

def fix_entries(entries, target_sec):
    if not entries:
        return [["Undefined", target_sec]]

    fixed = []

    for label, end_sec in entries:
        if end_sec < target_sec:
            fixed.append([label, end_sec])
        elif end_sec == target_sec:
            fixed.append([label, target_sec])
            return fixed
        else:
            fixed.append([label, target_sec])
            return fixed

    if fixed[-1][1] < target_sec:
        fixed.append(["Undefined", target_sec])

    return fixed

# Basic usability filters
df["preprocessed_exists"] = df["file_path_preprocessed_signals"].map(lambda p: Path(p).exists())
df["manual_exists"] = df["file_path_manual_state_annotation"].map(lambda p: Path(p).exists())

if "label_quality" not in df.columns:
    df["label_quality"] = "unknown"

# Pool: usable WT segments with manual labels, from week 2 and week 21
pool = df[
    (df["group"] == "WT")
    & (df["week"].isin([2, 21]))
    & as_bool(df["has_manual_labels"])
    & (df["pct_scored"] >= 0.90)
    & df["preprocessed_exists"]
    & df["manual_exists"]
].copy()

# Exclude explicitly uncertain/bad labels if present.
pool = pool[
    ~pool["label_quality"].astype(str).str.lower().isin(["uncertain", "bad", "exclude"])
].copy()

if len(pool) == 0:
    raise SystemExit("No usable WT manually scored segments found.")

print("Initial WT LOMO pool:")
print("Rows:", len(pool))
print("Mice:", sorted(pool["mouse_id"].unique()))
print("\nBy mouse/week:")
print(pool.groupby(["mouse_id", "week"]).size())
print("\nLabel quality distribution:")
print(pool["label_quality"].value_counts(dropna=False))

# Create length-fixed manual annotations for all selected WT segments
fixed_paths = []

print("\nCreating/checking length-fixed annotations...")

for i, row in pool.iterrows():
    signal_path = Path(row["file_path_preprocessed_signals"])
    manual_path = Path(row["file_path_manual_state_annotation"])

    n_epochs = len(np.load(signal_path))
    target_sec = n_epochs * EPOCH_SEC

    header2, entries = read_stage_duration(manual_path)
    old_sec = entries[-1][1] if entries else 0
    old_epochs = old_sec / EPOCH_SEC
    diff_epochs = n_epochs - old_epochs

    fixed_entries = fix_entries(entries, target_sec)
    fixed_path = manual_path.with_name("somnotate_annotation_lenfixed_lomo_512hz.tsv")
    write_stage_duration(fixed_path, target_sec, header2, fixed_entries)

    fixed_paths.append(str(fixed_path))

    status = "OK" if diff_epochs == 0 else "FIXED"
    print(
        f"{status} | mouse {row['mouse_id']} | week {row['week']} | "
        f"segment {row['segment_id']} | signal_epochs={n_epochs} | "
        f"annotation_epochs={old_epochs:.0f} | diff={diff_epochs:.0f}"
    )

pool["file_path_manual_state_annotation_original"] = pool["file_path_manual_state_annotation"]
pool["file_path_manual_state_annotation"] = fixed_paths

pool_out = OUT_DIR / "all_WT_labeled_pool_512hz_lenfixed.csv"
pool.to_csv(pool_out, index=False)

# Create one fold per WT mouse.
mice = sorted(pool["mouse_id"].unique())

for heldout_mouse in mice:
    train = pool[pool["mouse_id"] != heldout_mouse].copy()
    test = pool[pool["mouse_id"] == heldout_mouse].copy()

    suffix = f"lomo_holdout_mouse{heldout_mouse}"

    # Unique output paths for this fold, so we do not overwrite previous pilot results.
    for idx, row in test.iterrows():
        folder = Path(row["file_path_preprocessed_signals"]).parent
        test.at[idx, "file_path_automated_state_annotation"] = str(folder / f"somnotate_automated_{suffix}.tsv")
        test.at[idx, "file_path_review_intervals"] = str(folder / f"somnotate_review_intervals_{suffix}.csv")
        test.at[idx, "file_path_state_probabilities"] = str(folder / f"somnotate_state_probabilities_{suffix}.npz")

    train_out = OUT_DIR / f"fold_mouse{heldout_mouse}_train.csv"
    test_out = OUT_DIR / f"fold_mouse{heldout_mouse}_test.csv"

    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)

    print(f"\nFold mouse {heldout_mouse}:")
    print("  train segments:", len(train), "| train mice:", sorted(train["mouse_id"].unique()))
    print("  test segments:", len(test), "| test mouse:", heldout_mouse)
    print("  wrote:", train_out)
    print("  wrote:", test_out)

print("\nDone.")
print("Output folder:", OUT_DIR)
