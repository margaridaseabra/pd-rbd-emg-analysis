from pathlib import Path

OUT_DIR = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/final_WT_reference_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

out = OUT_DIR / "PD_inference_scoring_statement.txt"

text = """
PD INFERENCE / SCORING STATEMENT

The final sleep-state models used for downstream RBD-like event analysis were trained only on WT reference data.
PD recordings were not used to train these final reference models.

Therefore, model outputs on PD recordings should be described as:
- inference outputs,
- automated scoring,
- or WT-reference model scores.

They should not be described as unbiased validation accuracy on PD.

Model validation was estimated separately using leave-one-WT-mouse-out validation.
The PD outputs are used to quantify whether PD recordings contain EMG bursts occurring during EEG-defined REM-like states,
and whether those bursts show altered probability structure compared with WT recordings.

Recommended wording:
'A WT-trained reference model was applied consistently to WT and PD recordings. Model performance was estimated using leave-one-WT-mouse-out validation. PD recordings were treated as inference/scoring data for downstream RBD-like event analysis, not as an independent classifier-validation set.'
"""

out.write_text(text.strip() + "\n")

print("Wrote:")
print(out)
