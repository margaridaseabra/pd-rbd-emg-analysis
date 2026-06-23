# New week 5/8/10 analysis runbook

This document describes what to do when week 5, week 8, and week 10 data become available.

## Principle

Keep the week 2/week 21 pipeline fixed and apply the same pipeline to the new weeks.

Do not change EMG detection thresholds, morphology features, probability criteria, or QC labels unless the change is documented.

## Step 1: Fill the manifest

Start from:

    data/manifests/new_weeks_manifest_template.csv

For each recording, fill:

- mouse_id
- group: WT or PD
- week: 5, 8, or 10
- segment_id
- recording_name
- raw_signal_path
- manual_scores_path
- eegonly_prob_path
- full_prob_path
- video_path, if available
- exp_path, if available
- notes

Raw data should stay outside GitHub.

## Step 2: Validate the manifest

Run:

    python Code/validate_new_week_manifest.py data/manifests/new_weeks_manifest_template.csv

Fix missing paths or wrong metadata before running the pipeline.

## Step 3: Run/locate state probabilities

For every new recording, make sure both probability outputs exist:

- EEG-only model probabilities
- Full EEG+EMG model probabilities

Both are needed to compute the REM dissociation metric:

    delta_REM = P_REM_EEGonly - P_REM_FULL

## Step 4: Run EMG burst detection

Use the same settings as week 2/week 21:

- RMS window: 0.25 s
- onset threshold: z >= 4
- offset threshold: z < 2
- microburst merge gap: 0.5 s
- episode merge gap: 10 s
- minimum duration: 0.10 s
- baseline: quiet/high-confidence NREM

## Step 5: Categorize events

Use EEG-only and full-model probabilities to assign event classes such as:

- stable_REM_EMG_burst
- EMG_suppressed_REM
- mixed_REM_Wake_transition
- wake_like_movement
- NREM_like_EMG
- other_uncertain

## Step 6: Extract morphology features

Run the same morphology feature extraction used for week 2/week 21.

Do not change the feature list unless the analysis is explicitly restarted.

## Step 7: Assign morphology labels

Primary analysis:

- project new events into the existing morphology framework
- keep cluster definitions comparable with week 2/week 21

Secondary/exploratory analysis:

- refit UMAP/GMM with all weeks only after the fixed-framework analysis is complete

## Step 8: Add to the QC app

Create an all-weeks event table containing:

- week 2
- week 5
- week 8
- week 10
- week 21

The app should automatically show all available weeks from the event table.

## Step 9: QC priority

Prioritize QC in this order:

1. dissociation-positive events
2. REM-relevant automatic events
3. sustained/tonic events in REM-like periods
4. clustered high-tone events in REM-like periods
5. high-amplitude EMG events
6. events near REM exits/transitions

## Step 10: Main longitudinal metrics

Main mouse-level metrics:

- EMG events per EEG-only REM minute
- candidate EMG-suppressed REM per EEG-only REM minute
- stable REM EMG bursts per stable REM minute
- sustained/tonic REM events per EEG-only REM minute
- clustered high-tone REM events per EEG-only REM minute
- mean delta_REM during REM-relevant EMG events
- REM fragmentation index
- QC-confirmed possible RBD-like events per EEG-only REM minute

## Step 11: Main plots

Use longitudinal mouse-level plots:

- x-axis: week 2, 5, 8, 10, 21
- lines: individual mice
- color: group, WT vs PD
- overlay group mean if useful

Avoid relying only on event-level counts.

## Step 12: Git workflow

Work on the branch:

    add-week5-week8-week10

Commit after each safe step:

    git add .
    git commit -m "Describe the completed step"
    git push

