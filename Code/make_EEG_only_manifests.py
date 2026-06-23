from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
# Set this after checking state_annotation_signals.
# If EEG was first in the preprocessing config, use "first".
# If EEG was second, use "second".
EEG_FEATURE_BLOCK = "first"   # "first" or "second"

MANIFEST_DIR = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")

WT_POOL = MANIFEST_DIR / "wt_lomo_validation" / "all_WT_labeled_pool_512hz_lenfixed.csv"

PD_FINAL = MANIFEST_DIR / "final_WT_model_inference" / "PD_week2_week21_inference_finalWT_512hz_completed.csv"
if not PD_FINAL.exists():
    PD_FINAL = MANIFEST_DIR / "final_WT_model_inference" / "PD_week2_week21_inference_finalWT_512hz.csv"

OUT_DIR = MANIFEST_DIR / "EEG_only"
OUT_DIR.mkdir(exist_ok=True)

def eeg_only_path(path):
    path = Path(path)
    return path.with_name(path.stem + "_EEGonly" + path.suffix)

def output_suffix_path(path):
    path = Path(path)
    return path.with_name(path.stem + "_EEGonly" + path.suffix)

def make_eeg_only_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    out = eeg_only_path(path)

    if out.exists():
        arr = np.load(out)
        return out, arr.shape

    arr = np.load(path)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}: {path}")

    n_features = arr.shape[1]

    if n_features % 2 != 0:
        raise ValueError(
            f"Feature dimension is not divisible by 2, cannot split EEG/EMG cleanly: "
            f"{path}, shape={arr.shape}"
        )

    half = n_features // 2

    if EEG_FEATURE_BLOCK == "first":
        eeg = arr[:, :half]
    elif EEG_FEATURE_BLOCK == "second":
        eeg = arr[:, half:]
    else:
        raise ValueError("EEG_FEATURE_BLOCK must be 'first' or 'second'.")

    np.save(out, eeg)
    return out, eeg.shape

def convert_manifest(in_path, out_path, keep_only_existing=True):
    df = pd.read_csv(in_path)
    rows = []

    print(f"\nConverting manifest:\n  {in_path}")

    for i, row in df.iterrows():
        try:
            old_pre = Path(row["file_path_preprocessed_signals"])
            new_pre, shape = make_eeg_only_file(old_pre)

            row = row.copy()
            row["file_path_preprocessed_signals_original_EEG_EMG"] = str(old_pre)
            row["file_path_preprocessed_signals"] = str(new_pre)
            row["n_epochs_EEGonly"] = shape[0]
            row["n_features_EEGonly"] = shape[1]

            # For inference manifests, avoid overwriting old EEG+EMG outputs.
            for col in [
                "file_path_automated_state_annotation",
                "file_path_review_intervals",
                "file_path_state_probabilities",
            ]:
                if col in row and isinstance(row[col], str) and row[col]:
                    row[col] = str(output_suffix_path(row[col]))

            rows.append(row)

            print(
                f"{i}: OK | {row.get('recording_name', '')} | "
                f"mouse {row.get('mouse_id', '')} | segment {row.get('segment_id', '')} | "
                f"shape={shape}"
            )

        except Exception as e:
            print(
                f"{i}: SKIP | {row.get('recording_name', '')} | "
                f"segment {row.get('segment_id', '')} | error={repr(e)}"
            )

            if not keep_only_existing:
                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print("\nWrote:")
    print(" ", out_path)
    print("Rows:", len(out_df))

    if len(out_df):
        print("\nFeature dimensions:")
        print(out_df["n_features_EEGonly"].value_counts())

    return out_df

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if not WT_POOL.exists():
    raise FileNotFoundError(f"Could not find WT pool manifest: {WT_POOL}")

wt_out = OUT_DIR / "all_WT_labeled_pool_512hz_lenfixed_EEGonly.csv"
convert_manifest(WT_POOL, wt_out)

if PD_FINAL.exists():
    pd_out = OUT_DIR / "PD_week2_week21_inference_finalWT_512hz_EEGonly.csv"
    convert_manifest(PD_FINAL, pd_out)
else:
    print("\nNo PD final inference manifest found, skipping PD EEG-only manifest.")

print("\nDone.")
print("Output folder:")
print(OUT_DIR)
