# Week 5 / 8 / 10 integration plan

Principle: keep the current model, detector, feature definitions, and morphology atlas fixed for the first pass. Add the new weeks as application/inference data so they are comparable to week 2 and week 21.

Steps:

1. Create an all-weeks manifest with week 2, 5, 8, 10, and 21.
2. Run EEG-only and FULL EEG+EMG inference for new weeks using the same reference model.
3. Run the same EMG event detector with the same parameters.
4. Compute REM/Wake dissociation and event categories.
5. Extract the same morphology features.
6. Assign new events to the existing morphology model/atlas first; refit only as a secondary check.
7. Merge into an all-weeks QC table using stable event keys.
8. QC a balanced subset of REM-relevant, dissociation-positive, sustained/tonic, and clustered/high-tone events.
9. Compute mouse-level, REM-normalized longitudinal metrics.
