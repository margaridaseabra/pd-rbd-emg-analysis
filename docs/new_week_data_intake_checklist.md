# New week data intake checklist

Use this checklist when adding new week 5, week 8, or week 10 recordings.

## Required information per recording

- Mouse ID
- Group: WT or PD
- Week: 5, 8, or 10
- Segment ID
- Recording name
- Raw EEG/EMG signal file path
- Manual sleep scoring file path, if available
- EEG-only state probability file path
- Full EEG+EMG state probability file path
- Video file path, if available
- EXP/video mapping file path, if available
- Notes about missing files, bad channels, or unusual recording problems

## Processing order

1. Add recording to manifest.
2. Confirm raw signal and scoring files can be loaded.
3. Run or locate EEG-only probabilities.
4. Run or locate full EEG+EMG probabilities.
5. Run EMG burst detection using the same settings as week 2/21.
6. Categorize EMG events using EEG-only and full probabilities.
7. Extract morphology features.
8. Assign events to the existing morphology framework.
9. Add events to QC app.
10. Compute mouse-level REM-normalized metrics.

## Important rule

Do not change detection thresholds, morphology features, or probability criteria when adding new weeks unless there is a documented reason.

The goal is to make week 5/8/10 directly comparable to week 2 and week 21.
