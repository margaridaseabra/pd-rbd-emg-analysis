# Longitudinal week hardcoding audit

This audit lists where the current code assumes only week 2 and week 21.

## Current status

The Streamlit app opens successfully and currently shows only week 2 and week 21 because the current event table only contains those weeks.

This is expected. When week 5, week 8, and week 10 are added to the event table, the app should show them automatically if filters are data-driven.

## Files that need future longitudinal updates

### emg_burst_interactive_QC_app.py

Contains W2/W21-specific exploratory statistics, especially contrasts such as:

- PD W21 vs WT W21
- PD W21 vs PD W2
- WT W21 vs WT W2
- PD W2 vs WT W2

These sections are useful for the current analysis but should later be generalized to support arbitrary weeks.

Priority: high.

### compare_supervised_RBD_metrics_groups.py

Contains hardcoded labels:

- WT W2
- WT W21
- PD W2
- PD W21

and paired plots only comparing week 2 to week 21.

This should be generalized before running final longitudinal statistics with week 5, 8, and 10.

Priority: high.

### Code/plot_somnotate_summary_outputs.py

Contains a fixed condition order for WT/PD week 2 and week 21.

Priority: medium.

### Code/evaluate_EEGonly_on_PD.py and Code/evaluate_finalWT_on_PD.py

Filter to weeks [2, 21]. These are likely validation/exploratory scripts.

Priority: medium/low.

### Code/make_wt_lomo_manifests.py

Uses WT week 2 and week 21 for reference/training validation.

This may be intentional and should not be changed unless retraining/validation design changes.

Priority: low.

## Legacy PD21 scripts

The following scripts are intentionally PD21-specific and do not need to be generalized now:

- Code/plot_PD21_custom_period.py
- Code/plot_PD21_custom_period_EEGonly.py
- Code/plot_PD21_interesting_REM_moments.py
- Code/plot_PD21_NREM_REM_WAKE_sequences.py
- Code/plot_PD21_manualWake_modelREM_moments.py
- Code/summarize_PD21_REM_bouts.py
- Code/make_PD21_gui_manifest.py

They can remain as legacy exploratory scripts.

## Rule for adding week 5, 8, and 10

Do not manually add fake week options to the app.

The app should read available weeks from the event table:

    sorted(events["week"].dropna().astype(int).unique())

Current table:
    [2, 21]

Future all-weeks table:
    [2, 5, 8, 10, 21]

## Next coding task

Generalize only the main app and main comparison script, while leaving PD21 legacy scripts unchanged.
