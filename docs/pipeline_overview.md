# Pipeline overview

1. Build/validate sleep-state model outputs.
2. Run EEG-only and EEG+EMG inference.
3. Detect EMG events using quiet NREM baseline thresholding.
4. Score each event with sleep-state probabilities and REM/Wake dissociation metrics.
5. Review events in the QC app.
6. Extract EMG morphology features.
7. Assign events to morphology classes / UMAP atlas.
8. Compute mouse-level REM-normalized burden metrics.
9. Compare genotype/week and longitudinal progression.
