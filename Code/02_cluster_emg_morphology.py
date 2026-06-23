from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "PD-Katia/Data/prepared_data/manifests"
)

DEFAULT_IN = BASE / "EMG_unsupervised_morphology" / "emg_morphology_features.csv"
DEFAULT_OUT_DIR = BASE / "EMG_unsupervised_morphology"


ID_COLS = [
    "qc_event_id",
    "stable_event_key",
    "recording_name",
    "group",
    "week",
    "mouse_id",
    "segment_id",
    "primary_category",
    "event_class",
    "manual_state_center",
    "EEGonly_state_center",
    "full_state_center",
    "qc_status",
    "qc_notes",
    "start_sec",
    "end_sec",
    "file_path_raw_signals",
    "emg_label",
    "feature_extraction_ok",
    "feature_extraction_error",
]


EXCLUDE_FEATURE_PATTERNS = [
    "qc_event_id",
    "mouse_id",
    "segment_id",
    "week",
    "fs",
]


def choose_feature_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = []

    for c in numeric_cols:
        if c in ID_COLS:
            continue

        if c in EXCLUDE_FEATURE_PATTERNS:
            continue

        # Keep morphology/probability/context columns, but avoid direct identifiers.
        if c in ["start_sec", "end_sec"]:
            continue

        # Remove columns that are almost entirely missing.
        if df[c].notna().mean() < 0.40:
            continue

        # Remove near-constant columns.
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.nunique(dropna=True) <= 1:
            continue

        feature_cols.append(c)

    return feature_cols


def suggest_cluster_labels(df):
    """
    Descriptive labels based on cluster-level feature profiles.
    These are not used for clustering; they are only interpretive.
    """
    out = df.copy()

    labels = {}

    for cluster, sub in out.groupby("emg_morphology_cluster"):
        dur = sub.get("morph_duration_sec", pd.Series(dtype=float)).median()
        active = sub.get("active_fraction_in_event", pd.Series(dtype=float)).median()
        n_sub = sub.get("n_subbursts", pd.Series(dtype=float)).median()
        peak_ratio = sub.get("rms_peak_to_mean_ratio", pd.Series(dtype=float)).median()
        bg = sub.get("background_rms_median", pd.Series(dtype=float)).median()
        bg_all = out.get("background_rms_median", pd.Series(dtype=float)).median()

        label = "mixed_or_uncertain_EMG"

        if pd.notna(dur) and dur <= 2.0 and pd.notna(peak_ratio) and peak_ratio >= 2.0:
            label = "brief_phasic_twitch_like"

        if pd.notna(n_sub) and n_sub >= 3 and pd.notna(dur) and dur <= 10:
            label = "clustered_phasic_bursts"

        if pd.notna(dur) and dur > 10 and pd.notna(active) and active >= 0.50:
            label = "sustained_tonic_like_EMG"

        if pd.notna(bg) and pd.notna(bg_all) and bg > bg_all * 1.5:
            if label == "mixed_or_uncertain_EMG":
                label = "high_background_tone_EMG"
            else:
                label = label + "_with_high_background_tone"

        labels[cluster] = label

    out["suggested_morphology_label"] = out["emg_morphology_cluster"].map(labels)

    return out, labels


def plot_pca(df, color_col, out_path, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    if color_col not in df.columns:
        ax.scatter(df["pca1"], df["pca2"], s=35, alpha=0.8)
    else:
        values = df[color_col].astype(str).fillna("NA")
        cats = sorted(values.unique())

        for cat in cats:
            sub = df[values == cat]
            ax.scatter(sub["pca1"], sub["pca2"], s=35, alpha=0.8, label=cat)

        if len(cats) <= 12:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cluster_feature_heatmap(cluster_z, out_path):
    if len(cluster_z) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.28 * cluster_z.shape[1]), 4.5))
    im = ax.imshow(cluster_z.values, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(cluster_z.shape[0]))
    ax.set_yticklabels([f"Cluster {i}" for i in cluster_z.index])

    ax.set_xticks(np.arange(cluster_z.shape[1]))
    ax.set_xticklabels(cluster_z.columns, rotation=90, fontsize=7)

    ax.set_title("Cluster feature profile, z-scored relative to all events")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(DEFAULT_IN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--force-k", type=int, default=None)
    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)

    if "feature_extraction_ok" in df.columns:
        df = df[df["feature_extraction_ok"].astype(str).isin(["True", "true", "1"])].copy()

    feature_cols = choose_feature_columns(df)

    if len(feature_cols) < 3:
        raise ValueError(f"Too few usable feature columns: {feature_cols}")

    print(f"Using {len(feature_cols)} features:")
    for c in feature_cols:
        print(f"  - {c}")

    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imp = imputer.fit_transform(X_raw)
    X = scaler.fit_transform(X_imp)

    # PCA for visualization and clustering input.
    n_components = min(10, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    # Choose number of clusters by BIC unless forced.
    candidate_rows = []

    if args.force_k is not None:
        ks = [args.force_k]
    else:
        ks = list(range(args.min_k, min(args.max_k, len(df) - 1) + 1))

    best_model = None
    best_bic = np.inf
    best_k = None

    for k in ks:
        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=0,
                n_init=20,
            )
            model.fit(X_pca)

            labels = model.predict(X_pca)
            bic = model.bic(X_pca)

            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X_pca, labels)
            else:
                sil = np.nan

            candidate_rows.append({
                "k": k,
                "bic": bic,
                "silhouette": sil,
            })

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_k = k

        except Exception as e:
            candidate_rows.append({
                "k": k,
                "bic": np.nan,
                "silhouette": np.nan,
                "error": repr(e),
            })

    if best_model is None:
        raise RuntimeError("Could not fit any Gaussian mixture model.")

    labels = best_model.predict(X_pca)
    probs = best_model.predict_proba(X_pca)
    confidence = probs.max(axis=1)

    df["emg_morphology_cluster"] = labels
    df["emg_morphology_cluster_probability"] = confidence

    df, suggested_labels = suggest_cluster_labels(df)

    # Save model selection.
    model_selection = pd.DataFrame(candidate_rows)
    model_selection.to_csv(out_dir / "emg_morphology_gmm_model_selection.csv", index=False)

    # Save feature list.
    pd.DataFrame({"feature": feature_cols}).to_csv(
        out_dir / "emg_morphology_features_used_for_clustering.csv",
        index=False,
    )

    # Cluster summary.
    summary = (
        df.groupby(["emg_morphology_cluster", "suggested_morphology_label"], as_index=False)
        .agg(
            n_events=("qc_event_id", "count"),
            mean_cluster_probability=("emg_morphology_cluster_probability", "mean"),
            median_duration_sec=("morph_duration_sec", "median"),
            median_rms_z_max=("emg_rms_z_max", "median"),
            median_active_fraction=("active_fraction_in_event", "median"),
            median_n_subbursts=("n_subbursts", "median"),
            median_background_rms=("background_rms_median", "median"),
            median_delta_REM=("delta_REM", "median"),
            median_P_REM_EEGonly=("P_REM_EEGonly", "median"),
            median_P_REM_FULL=("P_REM_FULL", "median"),
        )
    )

    summary.to_csv(out_dir / "emg_morphology_cluster_summary.csv", index=False)

    # Cluster composition by group/week/category.
    for cols, name in [
        (["group", "week", "emg_morphology_cluster", "suggested_morphology_label"], "cluster_composition_by_group_week.csv"),
        (["primary_category", "emg_morphology_cluster", "suggested_morphology_label"], "cluster_composition_by_category.csv"),
        (["qc_status", "emg_morphology_cluster", "suggested_morphology_label"], "cluster_composition_by_qc_status.csv"),
    ]:
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) >= 2:
            comp = df.groupby(valid_cols, as_index=False).size()
            comp.to_csv(out_dir / name, index=False)

    # Cluster feature profile.
    cluster_means = df.groupby("emg_morphology_cluster")[feature_cols].mean()

    global_mean = df[feature_cols].mean()
    global_std = df[feature_cols].std().replace(0, np.nan)

    cluster_z = (cluster_means - global_mean) / global_std
    cluster_z.to_csv(out_dir / "emg_morphology_cluster_feature_zprofiles.csv")

    # Save final table.
    out_table = out_dir / "emg_morphology_features_with_clusters.csv"
    df.to_csv(out_table, index=False)

    # Plots.
    plot_pca(
        df,
        "emg_morphology_cluster",
        out_dir / "pca_emg_morphology_by_cluster.png",
        f"EMG morphology PCA, GMM k={best_k}",
    )

    for color_col in ["group", "week", "primary_category", "qc_status", "suggested_morphology_label"]:
        if color_col in df.columns:
            plot_pca(
                df,
                color_col,
                out_dir / f"pca_emg_morphology_by_{color_col}.png",
                f"EMG morphology PCA colored by {color_col}",
            )

    plot_cluster_feature_heatmap(
        cluster_z,
        out_dir / "emg_morphology_cluster_feature_heatmap.png",
    )

    print("\nBest GMM k:", best_k)
    print("Suggested cluster labels:")
    for k, label in suggested_labels.items():
        print(f"  Cluster {k}: {label}")

    print("\nSaved:")
    print(out_table)
    print(out_dir / "emg_morphology_cluster_summary.csv")
    print(out_dir / "pca_emg_morphology_by_cluster.png")
    print(out_dir / "emg_morphology_cluster_feature_heatmap.png")


if __name__ == "__main__":
    main()
