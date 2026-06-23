from pathlib import Path
import pandas as pd

CANDIDATES = [
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz_completed.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_model_inference/PD_week2_week21_inference_finalWT_512hz.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv"),
]

MANIFEST = next((p for p in CANDIDATES if p.exists()), None)

if MANIFEST is None:
    raise FileNotFoundError("Could not find a final or pilot inference manifest.")

df = pd.read_csv(MANIFEST)

pd21 = df[
    (df["group"] == "PD")
    & (df["week"] == 21)
].copy()

pd21["has_auto"] = pd21["file_path_automated_state_annotation"].map(lambda p: Path(p).exists())
pd21["has_probs"] = pd21["file_path_state_probabilities"].map(lambda p: Path(p).exists())
pd21["has_review"] = pd21["file_path_review_intervals"].map(lambda p: Path(p).exists())

pd21 = pd21[
    pd21["has_auto"]
    & pd21["has_probs"]
].copy()

OUT_DIR = MANIFEST.parent / "PD21_GUI"
OUT_DIR.mkdir(exist_ok=True)

out = OUT_DIR / "PD21_finalWT_for_GUI.csv"
pd21.to_csv(out, index=False)

print("Using source manifest:")
print(MANIFEST)
print()
print("Wrote PD week 21 GUI manifest:")
print(out)
print()
print("Rows:", len(pd21))
print()
print("Use these --only indices in the GUI:")
print(pd21[[
    "recording_name",
    "mouse_id",
    "week",
    "segment_id",
    "has_auto",
    "has_probs",
    "has_review",
]].reset_index(drop=True).to_string())
