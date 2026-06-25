#!/usr/bin/env python3
"""
Compute all-weeks EMG event morphology.

This script:
- reads the all-weeks QC-ready EMG event table
- builds morphology features from event-level EMG descriptors
- clusters events with GMM
- assigns interpretable morphology labels
- saves CSV summaries and PCA figures
- optionally updates the event table in-place so the Streamlit app can use the labels

Important:
The clustering does NOT use group, week, genotype, QC labels, REM probabilities,
or manual labels. This avoids leaking biological labels into morphology.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


EXCLUDE_TOKENS = [
    "group", "genotype", "sex", "mouse", "week", "segment",
    "state", "prob", "p_rem", "p_wake", "p_nrem", "delta",
    "qc", "label", "category", "class", "cluster",
    "video", "path", "file", "key", "note", "id",
    "start", "end", "time",
]

INCLUDE_TOKENS = [
    "duration", "micro", "burst", "emg", "rms", "auc", "area",
    "peak", "amplitude", "density", "gap", "interval", "rate",
    "z", "n_",
]


def is_candidate_feature(col: str) -> bool:
    low = col.lower()

    if any(tok in low for tok in EXCLUDE_TOKENS):
        return False

    return any(tok in low for tok in INCLUDE_TOKENS)


def find_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []

    for c in df.columns:
        if not is_candidate_feature(c):
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        valid_frac = x.notna().mean()
        nunique = x.nunique(dropna=True)

        if valid_frac >= 0.50 and nunique >= 3:
            cols.append(c)

    # Add safe known columns if present
    known = [
        "duration_sec_for_category",
        "duration_sec",
        "n_microbursts",
        "max_EMG_z",
        "max_EMG_baseline_z",
        "mean_EMG_z",
        "rms_EMG",
        "emg_rms",
        "emg_area",
        "emg_auc",
    ]

    for c in known:
        if c in df.columns and c not in cols:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() >= 0.50 and x.nunique(dropna=True) >= 3:
                cols.append(c)

    return cols


def prepare_features(df: pd.DataFrame, feature_cols: list[str]):
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # Drop very sparse columns
    keep = [c for c in X_raw.columns if X_raw[c].notna().mean() >= 0.50]
    X_raw = X_raw[keep]

    if X_raw.shape[1] < 2:
        raise ValueError(
            "Too few usable morphology features. Usable columns were: "
            + ", ".join(X_raw.columns)
        )

    # Median imputation
    X = X_raw.copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())

    # Log transform positive skewed features
    X_trans = X.copy()
    for c in X_trans.columns:
        x = X_trans[c]
        if x.min() >= 0:
            if any(tok in c.lower() for tok in ["duration", "micro", "burst", "rms", "area", "auc", "peak", "amplitude", "emg", "z"]):
                X_trans[c] = np.log1p(x)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trans)

    return X_raw, X, X_trans, X_scaled, list(X_trans.columns)


def choose_gmm(X_scaled: np.ndarray, out_dir: Path, random_state: int = 42):
    n = X_scaled.shape[0]
    max_k = min(8, max(2, n // 20))

    rows = []
    models = []

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
        )
        gmm.fit(X_scaled)

        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)

        rows.append({"n_components": k, "bic": bic, "aic": aic})
        models.append(gmm)

    model_selection = pd.DataFrame(rows)
    model_selection.to_csv(out_dir / "all_weeks_emg_morphology_gmm_model_selection.csv", index=False)

    best_idx = model_selection["bic"].idxmin()
    best_k = int(model_selection.loc[best_idx, "n_components"])
    best_model = models[best_idx]

    return best_model, model_selection, best_k


def first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def label_clusters(df: pd.DataFrame, cluster_col: str):
    cols = list(df.columns)

    duration_col = first_existing(cols, ["duration_sec_for_category", "duration_sec"])
    micro_col = first_existing(cols, ["n_microbursts", "n_microburst", "microburst_count"])
    max_z_col = first_existing(cols, ["max_EMG_z", "max_EMG_baseline_z", "max_emg_z"])

    summaries = []

    for cl, sub in df.groupby(cluster_col):
        duration = pd.to_numeric(sub[duration_col], errors="coerce").median() if duration_col else np.nan
        n_micro = pd.to_numeric(sub[micro_col], errors="coerce").median() if micro_col else np.nan
        max_z = pd.to_numeric(sub[max_z_col], errors="coerce").median() if max_z_col else np.nan

        if pd.isna(duration):
            fine = "mixed_uncertain"
            broad = "mixed"
        elif duration <= 0.35 and (pd.isna(max_z) or max_z < 6):
            fine = "ultra_brief_low_confidence"
            broad = "ultra_brief"
        elif duration <= 1.25 and (pd.isna(n_micro) or n_micro <= 1.5):
            fine = "brief_phasic_twitch_like"
            broad = "phasic"
        elif duration <= 3.0 and (pd.isna(n_micro) or n_micro <= 2.5):
            fine = "phasic_twitch_like"
            broad = "phasic"
        elif duration <= 8.0 and (not pd.isna(n_micro) and n_micro >= 2.0):
            fine = "clustered_phasic"
            broad = "clustered_phasic"
        elif duration > 8.0 and (pd.isna(n_micro) or n_micro <= 2.5):
            fine = "sustained_tonic_like"
            broad = "sustained_tonic"
        elif duration > 8.0 and (not pd.isna(n_micro) and n_micro > 2.5):
            fine = "clustered_sustained"
            broad = "sustained_tonic"
        else:
            fine = "mixed_intermediate"
            broad = "mixed"

        summaries.append({
            "emg_morphology_cluster": cl,
            "n_events": len(sub),
            "median_duration_sec": duration,
            "median_n_microbursts": n_micro,
            "median_max_EMG_z": max_z,
            "emg_morphology_fine_label": fine,
            "emg_morphology_broad_label": broad,
        })

    cluster_summary = pd.DataFrame(summaries)

    fine_map = dict(zip(cluster_summary["emg_morphology_cluster"], cluster_summary["emg_morphology_fine_label"]))
    broad_map = dict(zip(cluster_summary["emg_morphology_cluster"], cluster_summary["emg_morphology_broad_label"]))

    df["emg_morphology_fine_label"] = df[cluster_col].map(fine_map)
    df["emg_morphology_broad_label"] = df[cluster_col].map(broad_map)
    df["emg_morphology_class"] = df["emg_morphology_broad_label"]

    return df, cluster_summary


def make_pca_figures(df: pd.DataFrame, out_dir: Path):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def scatter_plot(color_col: str, filename: str):
        if color_col not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(6, 5))

        for value, sub in df.groupby(color_col, dropna=False):
            ax.scatter(
                sub["pca_1"],
                sub["pca_2"],
                s=8,
                alpha=0.45,
                label=str(value),
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(filename.replace("_", " ").replace(".png", ""))
        ax.legend(markerscale=2, fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=300)
        fig.savefig(fig_dir / filename.replace(".png", ".svg"))
        plt.close(fig)

    scatter_plot("emg_morphology_cluster", "pca_by_morphology_cluster.png")
    scatter_plot("emg_morphology_broad_label", "pca_by_morphology_broad_label.png")
    scatter_plot("primary_category", "pca_by_primary_category.png")
    scatter_plot("group", "pca_by_group.png")
    scatter_plot("week", "pca_by_week.png")

    # Composition by group/week
    if {"group", "week", "emg_morphology_broad_label"}.issubset(df.columns):
        comp = (
            df.groupby(["group", "week", "emg_morphology_broad_label"])
            .size()
            .reset_index(name="n_events")
        )
        comp.to_csv(out_dir / "all_weeks_emg_morphology_composition_by_group_week.csv", index=False)

        pivot = comp.pivot_table(
            index=["group", "week"],
            columns="emg_morphology_broad_label",
            values="n_events",
            fill_value=0,
        )

        frac = pivot.div(pivot.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(9, 5))
        frac.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("% of events")
        ax.set_xlabel("Group / week")
        ax.set_title("EMG morphology composition by group and week")
        ax.legend(title="Morphology", frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "morphology_composition_by_group_week.png", dpi=300)
        fig.savefig(fig_dir / "morphology_composition_by_group_week.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--update-events-in-place", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events, low_memory=False)
    events = events.copy()

    feature_cols = find_feature_columns(events)

    if len(feature_cols) < 2:
        raise RuntimeError(
            "Could not find enough morphology features in the event table.\n"
            "Found: " + ", ".join(feature_cols)
        )

    print("Using morphology features:")
    for c in feature_cols:
        print(" ", c)

    X_raw, X, X_trans, X_scaled, used_cols = prepare_features(events, feature_cols)

    pd.DataFrame({"feature": used_cols}).to_csv(
        args.out_dir / "all_weeks_emg_morphology_features_used.csv",
        index=False,
    )

    gmm, model_selection, best_k = choose_gmm(X_scaled, args.out_dir, args.random_state)

    clusters = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled).max(axis=1)

    pca = PCA(n_components=2, random_state=args.random_state)
    pca_xy = pca.fit_transform(X_scaled)

    events["emg_morphology_cluster"] = clusters.astype(int)
    events["emg_morphology_cluster_probability"] = probs
    events["pca_1"] = pca_xy[:, 0]
    events["pca_2"] = pca_xy[:, 1]

    events, cluster_summary = label_clusters(events, "emg_morphology_cluster")

    cluster_summary.to_csv(
        args.out_dir / "all_weeks_emg_morphology_cluster_summary.csv",
        index=False,
    )

    # Cluster feature medians and z-profiles
    raw_with_cluster = X.copy()
    raw_with_cluster["emg_morphology_cluster"] = events["emg_morphology_cluster"].values

    cluster_feature_medians = (
        raw_with_cluster.groupby("emg_morphology_cluster")
        .median(numeric_only=True)
        .reset_index()
    )
    cluster_feature_medians.to_csv(
        args.out_dir / "all_weeks_emg_morphology_cluster_feature_medians.csv",
        index=False,
    )

    features_with_clusters_path = args.out_dir / "all_weeks_emg_morphology_features_with_clusters.csv"
    events.to_csv(features_with_clusters_path, index=False)

    make_pca_figures(events, args.out_dir)

    if args.update_events_in_place:
        backup = args.events.with_name(args.events.stem + "_before_all_weeks_morphology_backup.csv")
        original = pd.read_csv(args.events, low_memory=False)
        original.to_csv(backup, index=False)

        events.to_csv(args.events, index=False)

        print("\nUpdated event table in-place:")
        print(args.events)
        print("Backup written to:")
        print(backup)

    print("\nBest GMM k:", best_k)
    print("\nWrote morphology table:")
    print(features_with_clusters_path)

    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))

    print("\nComposition by group/week:")
    print(pd.crosstab([events["group"], events["week"]], events["emg_morphology_broad_label"]).to_string())


if __name__ == "__main__":
    main()
