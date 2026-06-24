# QC review plan for all-weeks RBD/EMG analysis

## Goal

Review automatic EMG/RBD-like events across all groups and weeks so that final metrics can be computed from QC-confirmed events.

## Source table

`EMG_episodes_NREMbaseline_all_weeks_qc_ready.csv`

This is the source-of-truth event table used by the Streamlit QC app.

## Main QC labels

- `possible_RBD_like`: event likely reflects REM-compatible EMG activity
- `real_burst_but_wake`: real movement but wake-like
- `transition_event`: event near REM/wake transition
- `artifact`: signal/video artifact
- `uncertain`: unclear
- `exclude`: should not be used
- `not_reviewed`: not reviewed yet

## Priority order

1. Review all stable REM EMG burst events.
2. Review all EMG-suppressed REM events.
3. Review top mixed REM/wake transition events by RBD priority score or delta REM.
4. Review outlier mouse-weeks:
   - PD mouse 5 week 8
   - WT mouse 1 week 10
   - WT mouse 11 week 10
   - PD mouse 12 week 21
5. Review a random sample of wake-like/NREM-like/uncertain events as negative controls.

## Important rule

Do not treat unreviewed events as negative. They are unknown.

QC-confirmed metrics should use reviewed positive events as numerator and EEG-only REM minutes as denominator.
