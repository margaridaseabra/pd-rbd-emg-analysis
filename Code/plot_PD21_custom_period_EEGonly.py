from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse plotting/helper functions from the original full-model plotting script.
from plot_PD21_custom_period import (
    EPOCH_SEC,
    FMAX,
    STATE_ORDER,
    STATE_CMAP,
    load_probabilities,
    choose_manual_annotation_path,
    load_stage_duration,
    read_edf_window,
    robust_z,
    emg_rms,
    compute_spectrogram,
    state_bar,
)

EEGONLY_MANIFEST_CANDIDATES = [
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/EEG_only/PD21_EEGonly_for_GUI.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/EEG_only/PD_week2_week21_inference_finalWT_512hz_EEGonly_completed.csv"),
    Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/EEG_only/PD_week2_week21_inference_finalWT_512hz_EEGonly.csv"),
]

CUSTOM_OUT_DIR = Path.home() / "Desktop" / "PD21_custom_period_plots_EEGonly"
CUSTOM_OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x[:120]


def load_pd21_manifest():
    manifest = next((p for p in EEGONLY_MANIFEST_CANDIDATES if p.exists()), None)

    if manifest is None:
        raise FileNotFoundError("Could not find an EEG-only PD21 manifest.")

    df = pd.read_csv(manifest)

    pd21 = df[
        (df["group"] == "PD")
        & (df["week"] == 21)
    ].copy()

    pd21 = pd21[
        pd21["file_path_state_probabilities"].map(lambda p: Path(p).exists())
        & pd21["file_path_raw_signals"].map(lambda p: Path(p).exists())
    ].copy()

    pd21 = pd21.reset_index(drop=True)

    return manifest, pd21


def filter_rows(df, recording_contains=None, mouse_id=None, segment_id=None, row_index=None):
    out = df.copy()

    if row_index is not None:
        if row_index < 0 or row_index >= len(out):
            raise IndexError(f"--row-index {row_index} is out of range. Available rows: 0 to {len(out)-1}")
        return out.iloc[[row_index]].copy()

    if recording_contains:
        needle = recording_contains.lower()
        out = out[out["recording_name"].astype(str).str.lower().str.contains(needle, regex=False)].copy()

    if mouse_id is not None:
        out = out[out["mouse_id"].astype(int) == int(mouse_id)].copy()

    if segment_id is not None:
        out = out[out["segment_id"].astype(int) == int(segment_id)].copy()

    return out


def plot_custom_period(row, start_min, end_min, pad_min=2.0):
    prob_path = Path(row["file_path_state_probabilities"])
    raw_path = Path(row["file_path_raw_signals"])

    probs, state_names, pred, confidence, rem_prob = load_probabilities(prob_path)
    n_epochs = len(pred)

    manual = None
    manual_path = choose_manual_annotation_path(row.get("file_path_manual_state_annotation", ""))
    if manual_path is not None:
        try:
            manual = load_stage_duration(manual_path, n_epochs)
        except Exception:
            manual = None

    selected_start_s = float(start_min) * 60
    selected_end_s = float(end_min) * 60

    if selected_end_s <= selected_start_s:
        raise ValueError("end_min must be greater than start_min.")

    plot_start_s = max(0.0, selected_start_s - pad_min * 60)
    plot_end_s = selected_end_s + pad_min * 60

    t_sig, eeg, emg, fs, eeg_label, emg_label = read_edf_window(
        raw_path,
        plot_start_s,
        plot_end_s,
    )

    t_signal_min = t_sig / 60

    eeg_z = robust_z(eeg)
    emg_z = robust_z(emg)
    emg_rms_z = robust_z(emg_rms(emg, fs))

    f, t_spec, spec_img, vmin, vmax = compute_spectrogram(eeg, fs, fmax=FMAX)
    t_spec_min = (t_spec + plot_start_s) / 60

    epoch_start = max(0, int(np.floor(plot_start_s / EPOCH_SEC)))
    epoch_end = min(n_epochs, int(np.ceil(plot_end_s / EPOCH_SEC)))

    t_epoch_min = np.arange(epoch_start, epoch_end) * EPOCH_SEC / 60

    if len(t_epoch_min) == 0:
        max_min = n_epochs * EPOCH_SEC / 60
        raise ValueError(
            f"Selected period {start_min:.2f}-{end_min:.2f} min is outside "
            f"available EEG-only probability/state data. This segment only has "
            f"{max_min:.2f} min."
        )

    extent = [
        t_epoch_min[0],
        t_epoch_min[-1] + EPOCH_SEC / 60,
        0,
        1,
    ]

    has_manual = manual is not None

    if has_manual:
        fig, axes = plt.subplots(
            6, 1,
            figsize=(14, 11),
            gridspec_kw={"height_ratios": [0.55, 0.55, 1.5, 1.1, 1.1, 2.3]},
        )
    else:
        fig, axes = plt.subplots(
            5, 1,
            figsize=(14, 10),
            gridspec_kw={"height_ratios": [0.55, 1.5, 1.1, 1.1, 2.3]},
        )

    ax_i = 0

    if has_manual:
        axes[ax_i].imshow(
            state_bar(manual[epoch_start:epoch_end]),
            aspect="auto",
            interpolation="nearest",
            cmap=STATE_CMAP,
            vmin=0,
            vmax=len(STATE_ORDER) - 1,
            extent=extent,
        )
        axes[ax_i].set_yticks([])
        axes[ax_i].set_ylabel("Manual")
        ax_i += 1

    axes[ax_i].imshow(
        state_bar(pred[epoch_start:epoch_end]),
        aspect="auto",
        interpolation="nearest",
        cmap=STATE_CMAP,
        vmin=0,
        vmax=len(STATE_ORDER) - 1,
        extent=extent,
    )
    axes[ax_i].set_yticks([])
    axes[ax_i].set_ylabel("EEG-only\nSomnotate")
    axes[ax_i].text(
        1.01, 0.5,
        "Awake\nNREM\nREM\nUndefined",
        transform=axes[ax_i].transAxes,
        va="center",
        fontsize=8,
    )
    ax_i += 1

    for j, state in enumerate(state_names):
        axes[ax_i].plot(
            t_epoch_min,
            probs[epoch_start:epoch_end, j],
            label=state,
            linewidth=1.4,
        )

    axes[ax_i].plot(
        t_epoch_min,
        confidence[epoch_start:epoch_end],
        label="max prob",
        linewidth=1.0,
        linestyle="--",
    )
    axes[ax_i].set_ylim(-0.02, 1.02)
    axes[ax_i].set_ylabel("EEG-only\nprobability")
    axes[ax_i].legend(loc="upper right", fontsize=8)
    ax_i += 1

    axes[ax_i].plot(t_signal_min, eeg_z, linewidth=0.45)
    axes[ax_i].set_ylabel(f"EEG\n{eeg_label}\nrobust z")
    ax_i += 1

    axes[ax_i].plot(t_signal_min, emg_z, linewidth=0.35, alpha=0.65, label="EMG raw")
    axes[ax_i].plot(t_signal_min, emg_rms_z, linewidth=1.0, label="EMG RMS")
    axes[ax_i].set_ylabel(f"EMG\n{emg_label}\nrobust z")
    axes[ax_i].legend(loc="upper right", fontsize=8)
    ax_i += 1

    axes[ax_i].imshow(
        spec_img,
        origin="lower",
        aspect="auto",
        extent=[t_spec_min.min(), t_spec_min.max(), f.min(), f.max()],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[ax_i].set_ylim(0, FMAX)
    axes[ax_i].set_ylabel("EEG\nHz")
    axes[ax_i].set_xlabel("Time from segment start (min)")

    for ax in axes:
        ax.axvspan(start_min, end_min, alpha=0.16)
        ax.axvline(start_min, linestyle="--", linewidth=1)
        ax.axvline(end_min, linestyle="--", linewidth=1)
        ax.set_xlim(plot_start_s / 60, plot_end_s / 60)

    selected_epochs = np.arange(
        max(0, int(np.floor(selected_start_s / EPOCH_SEC))),
        min(n_epochs, int(np.ceil(selected_end_s / EPOCH_SEC))),
    )

    if len(selected_epochs):
        pred_counts = pd.Series(pred[selected_epochs]).value_counts(normalize=True) * 100
        rem_prob_mean = float(np.mean(rem_prob[selected_epochs]))
        conf_mean = float(np.mean(confidence[selected_epochs]))
    else:
        pred_counts = pd.Series(dtype=float)
        rem_prob_mean = np.nan
        conf_mean = np.nan

    pred_summary = ", ".join([f"{k}: {v:.1f}%" for k, v in pred_counts.items()])

    title = (
        f"EEG-only custom PD week 21 period | mouse {row['mouse_id']} | segment {row['segment_id']} | "
        f"{start_min:.2f}–{end_min:.2f} min\n"
        f"EEG-only predicted states in selected period: {pred_summary} | "
        f"mean REM prob={rem_prob_mean:.2f}, mean confidence={conf_mean:.2f} | "
        f"{row['recording_name']}"
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = CUSTOM_OUT_DIR / (
        f"EEGonly_custom_PD21_mouse{row['mouse_id']}_seg{row['segment_id']}_"
        f"{start_min:.1f}_to_{end_min:.1f}min_{safe_name(row['recording_name'])}.png"
    )

    plt.savefig(out, dpi=180)
    plt.close()

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list", action="store_true", help="List available EEG-only PD week 21 rows and exit.")
    parser.add_argument("--row-index", type=int, default=None, help="Use row index from the listed PD21 table.")
    parser.add_argument("--recording-contains", default=None, help="Substring of recording_name to select.")
    parser.add_argument("--mouse-id", type=int, default=None, help="Mouse ID to select.")
    parser.add_argument("--segment-id", type=int, default=None, help="Segment ID to select.")
    parser.add_argument("--start-min", type=float, default=None, help="Start minute from segment start.")
    parser.add_argument("--end-min", type=float, default=None, help="End minute from segment start.")
    parser.add_argument("--pad-min", type=float, default=2.0, help="Minutes to show before and after selected period.")

    args = parser.parse_args()

    manifest, pd21 = load_pd21_manifest()

    print("Using EEG-only manifest:")
    print(manifest)
    print()

    available_cols = ["recording_name", "mouse_id", "segment_id", "file_path_state_probabilities"]
    print("Available EEG-only PD week 21 rows:")
    print(pd21[available_cols].reset_index().to_string(index=False))

    if args.list:
        return

    if args.start_min is None or args.end_min is None:
        raise SystemExit("Please provide --start-min and --end-min.")

    matches = filter_rows(
        pd21,
        recording_contains=args.recording_contains,
        mouse_id=args.mouse_id,
        segment_id=args.segment_id,
        row_index=args.row_index,
    )

    if len(matches) == 0:
        raise SystemExit("No matching EEG-only PD week 21 recording found. Use --list to see available rows.")

    if len(matches) > 1:
        print("\nMore than one row matched. Please narrow selection using --row-index, --mouse-id, or --segment-id.")
        print(matches[["recording_name", "mouse_id", "segment_id"]].reset_index().to_string(index=False))
        raise SystemExit()

    row = matches.iloc[0]

    print("\nSelected:")
    print(row[["recording_name", "mouse_id", "segment_id"]].to_string())
    print()

    out = plot_custom_period(
        row=row,
        start_min=args.start_min,
        end_min=args.end_min,
        pad_min=args.pad_min,
    )

    print("Wrote:")
    print(out)
    print()
    print("Open folder:")
    print(CUSTOM_OUT_DIR)


if __name__ == "__main__":
    main()
