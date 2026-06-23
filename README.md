# PD/RBD EMG Morphology and QC Analysis

This repository contains the working analysis code for EMG event detection, REM/Wake dissociation metrics, interactive QC, and unsupervised EMG morphology analysis in the PD/RBD mouse sleep project.

## Main entry points

- `emg_burst_interactive_QC_app.py` — Streamlit app for reviewing EMG episodes with EEG/EMG traces, spectrograms, model probabilities, QC labels, and optional video.
- `supervisor_unsupervised_results_app.py` — simplified results app for supervisor/presentation views.
- `detect_EMG_bursts_NREM_baseline.py` — EMG event detection using NREM-baseline thresholding.
- `prepare_NREM_baseline_episodes_for_QC.py` — prepares detected episodes for QC.
- `summarize_supervised_EMG_burst_metrics.py` — mouse-level REM quality and EMG/RBD-like burden metrics.
- `Code/01_extract_emg_morphology_features.py` to `Code/06_enhance_umap_with_qc_and_dissociation.py` — unsupervised morphology pipeline.

## Setup

```bash
conda create -n sleep_app python=3.11
conda activate sleep_app
pip install -r requirements.txt
```

## Running the main QC app

```bash
streamlit run emg_burst_interactive_QC_app.py
```

## Data policy

Raw EDF/video files, generated plots, cache folders, and QC output CSVs are intentionally excluded from git. Keep them in `data/` or external storage and point to them with a local config file.

## Suggested next development branch

```bash
git checkout -b add-week5-week8-week10
```

Then add the week 5/8/10 manifests and run the same frozen inference + EMG detection + morphology assignment pipeline.
