from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "PD-Katia/Data/prepared_data/manifests"
)

DEFAULT_IN = BASE / "EMG_unsupervised_morphology" / "emg_morphology_features.csv"
DEFAULT_OUT_DIR = BASE / "EMG_unsupervised_morphology" / "refined_morphology_clustering"


MORPHOLOGY_FEATURES = [
    # duration
    "morph_duration_sec",

    # amplitude
    "emg_abs_z_mean",
    "emg_abs_z_max",
    "emg_abs_z_p95",
    "emg_rms_z_mean",
    "emg_rms_z_max",
    "emg_rms_z_p95",
    "emg_rms_auc",
    "emg_event_to_background_rms_ratio",

    # background tone
    "background_rms_median",
    "background_rms_scale",
    "pre_event_rms_median",
    "post_event_rms_median",
    "pre_post_tone_change",

    # burstiness
    "n_subbursts",
    "subburst_rate_per_sec",
    "total_active_duration_sec",
    "active_fraction_in_event",
    "mean_subburst_duration_sec",
    "max_subburst_duration_sec",
    "n_rms_peaks",
    "rms_peak_rate_per_sec",

    # regularity
    "rms_peak_interval_mean_sec",
    "rms_peak_interval_std_sec",
    "rms_peak_interval_cv",
    "rms_peak_regularity_score",

    # shape
    "rms_peak_time_fraction",
    "rms_rise_time_sec",
    "rms_decay_time_sec",
    "rms_rise_decay_ratio",
    "rms_peak_to_mean_ratio",
    "rms_skewness",
    "rms_kurtosis",

    # EMG spectral shape
    "emg_spectral_centroid_hz",
    "emg_spectral_median_hz",
    "emg_spectral_high_fraction_50_150",
]


LOG1P_FEATURES = [
    "morph_duration_sec",
    "emg_abs_z_mean",
    "emg_abs_z_max",
    "emg_abs_z_p95",
    "emg_rms_z_mean",
    "emg_rms_z_max",
    "emg_rms_z_p95",
    "emg_rms_auc",
    "emg_event_to_background_rms_ratio",
    "background_rms_median",
    "background_rms_scale",
    "pre_event_rms_median",
    "post_event_rms_median",
    "n_subbursts",
    "subburst_rate_per_sec",
    "total_active_duration_sec",
    "mean_subburst_duration_sec",
    "max_subburst_duration_sec",
    "n_rms_peaks",
    "rms_peak_rate_per_sec",
    "rms_peak_interval_mean_sec",
    "rms_peak_interval_std_sec",
    "rms_rise_time_sec",
    "rms_decay_time_sec",
    "rms_peak_to_mean_ratio",
    "emg_spectral_centroid_hz",
    "emg_spectral_median_hz",
]


def winsorize_series(x, lo=0.01, hi=0.99):
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() < 5:
        return x
    qlo = x.quantile(lo)
    qhi = x.quantile(hi)
    return x.clip(qlo, qhi)


def prepare_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()

    # Fix active fraction small numerical overshoots.
    if "active_fraction_in_event" in X.columns:
        X["active_fraction_in_event"] = pd.to_numeric(
            X["active_fraction_in_event"], errors="coerce"
        ).clip(lower=0, upper=1)

    # Log-transform strongly positive/skewed features.
    for c in LOG1P_FEATURES:
        if c in X.columns:
            vals = pd.to_numeric(X[c], errors="coerce")
            vals = vals.clip(lower=0)
            X[c] = np.log1p(vals)

    # Winsorize after log transform.
    for c in X.columns:
        X[c] = winsorize_series(X[c])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    return X, X_scaled


def choose_k_by_bic_with_min_size(X_pca, min_k, max_k, min_cluster_size):
    rows = []
    best = None

    for k in range(min_k, max_k + 1):
        if k >= len(X_pca):
            continue

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
            n_init=30,
        )

        model.fit(X_pca)
        labels = model.predict(X_pca)

        counts = pd.Series(labels).value_counts().sort_index()
        bic = model.bic(X_pca)

        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_pca, labels)
        else:
            sil = np.nan

        valid_min_size = counts.min() >= min_cluster_size

        rows.append({
            "k": k,
            "bic": bic,
            "silhouette": sil,
            "min_cluster_size": int(counts.min()),
            "valid_min_size": bool(valid_min_size),
        })

        if valid_min_size:
            if best is None or bic < best["bic"]:
                best = {
                    "k": k,
                    "bic": bic,
                    "model": model,
                    "labels": labels,
                    "silhouette": sil,
                }

    # If no k satisfies min cluster size, use the lowest valid k.
    if best is None:
        model = GaussianMixture(
            n_components=min_k,
            covariance_type="full",
            random_state=0,
            n_init=30,
        )
        model.fit(X_pca)
        labels = model.predict(X_pca)

        best = {
            "k": min_k,
            "bic": model.bic(X_pca),
            "model": model,
            "labels": labels,
            "silhouette": silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else np.nan,
        }

    return best, pd.DataFrame(rows)


def describe_clusters(df, feature_cols):
    rows = []

    for cluster, sub in df.groupby("emg_morphology_cluster"):
        dur = sub["morph_duration_sec"].median() if "morph_duration_sec" in sub else np.nan
        amp = sub["emg_rms_z_max"].median() if "emg_rms_z_max" in sub else np.nan
        active = sub["active_fraction_in_event"].median() if "active_fraction_in_event" in sub else np.nan
        n_sub = sub["n_subbursts"].median() if "n_subbursts" in sub else np.nan
        peak_ratio = sub["rms_peak_to_mean_ratio"].median() if "rms_peak_to_mean_ratio" in sub else np.nan
        bg = sub["background_rms_median"].median() if "background_rms_median" in sub else np.nan

        label = "mixed_or_uncertain"

        if pd.notna(dur) and dur <= 2.0 and pd.notna(peak_ratio) and peak_ratio >= 1.5:
            label = "brief_phasic_twitch_like"

        if pd.notna(n_sub) and n_sub >= 2 and pd.notna(dur) and dur <= 10:
            label = "clustered_phasic_bursts"

        if pd.notna(dur) and dur > 5 and pd.notna(active) and active >= 0.5:
            label = "sustained_or_tonic_like"

        if pd.notna(bg):
            all_bg = df["background_rms_median"].median()
            if pd.notna(all_bg) and bg > 1.5 * all_bg:
                label = label + "_high_background_tone"

        rows.append({
            "emg_morphology_cluster": cluster,
            "suggested_morphology_label": label,
            "n_events": len(sub),
            "median_duration_sec": dur,
            "median_emg_rms_z_max": amp,
            "median_active_fraction": active,
            "median_n_subbursts": n_sub,
            "median_peak_to_mean_ratio": peak_ratio,
            "median_background_rms": bg,
        })

    label_df = pd.DataFrame(rows)
    mapping = dict(zip(label_df["emg_morphology_cluster"], label_df["suggested_morphology_label"]))

    df = df.copy()
    df["suggested_morphology_label"] = df["emg_morphology_cluster"].map(mapping)

    return df, label_df


def plot_pca(df, color_col, out_path, title, label_outliers=False):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    if color_col not in df.columns:
        ax.scatter(df["pca1"], df["pca2"], s=35, alpha=0.85)
    else:
        vals = df[color_col].astype(str).fillna("NA")
        for val in sorted(vals.unique()):
            sub = df[vals == val]
            ax.scatter(sub["pca1"], sub["pca2"], s=38, alpha=0.85, label=val)

        if vals.nunique() <= 12:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    if label_outliers and "qc_event_id" in df.columns:
        x = df["pca1"]
        y = df["pca2"]
        dist = np.sqrt(((x - x.median()) / (x.std() + 1e-9)) ** 2 + ((y - y.median()) / (y.std() + 1e-9)) ** 2)
        out = df.loc[dist.sort_values(ascending=False).head(10).index]
        for _, r in out.iterrows():
            ax.text(r["pca1"], r["pca2"], str(int(r["qc_event_id"])), fontsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pca_zoom(df, out_path):
    if len(df) < 5:
        return

    xlo, xhi = df["pca1"].quantile([0.02, 0.98])
    ylo, yhi = df["pca2"].quantile([0.02, 0.98])

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    vals = df["emg_morphology_cluster"].astype(str)
    for val in sorted(vals.unique()):
        sub = df[vals == val]
        ax.scatter(sub["pca1"], sub["pca2"], s=38, alpha=0.85, label=val)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("EMG morphology PCA zoomed to central 96%")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_heatmap(df, feature_cols, out_path):
    cluster_means = df.groupby("emg_morphology_cluster")[feature_cols].mean()
    global_mean = df[feature_cols].mean()
    global_std = df[feature_cols].std().replace(0, np.nan)
    z = (cluster_means - global_mean) / global_std
    z = z.clip(-3, 3)

    fig, ax = plt.subplots(figsize=(max(9, 0.32 * len(feature_cols)), 4.5))
    im = ax.imshow(z.values, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(z.shape[0]))
    ax.set_yticklabels([f"Cluster {i}" for i in z.index])

    ax.set_xticks(np.arange(z.shape[1]))
    ax.set_xticklabels(z.columns, rotation=90, fontsize=7)

    ax.set_title("Refined cluster morphology profiles")
    fig.colorbar(im, ax=ax, label="z-score clipped ±3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    z.to_csv(out_path.with_suffix(".csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(DEFAULT_IN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-duration-sec", type=float, default=60.0)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--force-k", type=int, default=None)
    parser.add_argument(
        "--rem-candidates-only",
        action="store_true",
        help="Use only REM-relevant candidate categories for RBD-like morphology clustering.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)

    if "feature_extraction_ok" in df.columns:
        df = df[df["feature_extraction_ok"].astype(str).isin(["True", "true", "1"])].copy()

    # Remove very long episodes from burst morphology clustering.
    if "morph_duration_sec" in df.columns:
        df = df[pd.to_numeric(df["morph_duration_sec"], errors="coerce") <= args.max_duration_sec].copy()

    if args.rem_candidates_only and "primary_category" in df.columns:
        keep_categories = [
            "stable_REM_EMG_burst",
            "EMG_suppressed_REM",
            "mixed_REM_Wake_transition",
            "other_uncertain",
        ]
        df = df[df["primary_category"].isin(keep_categories)].copy()

    if len(df) < 10:
        raise ValueError(f"Too few events after filtering: {len(df)}")

    feature_cols = [c for c in MORPHOLOGY_FEATURES if c in df.columns]

    # Drop columns with too much missingness or no variation.
    final_features = []
    for c in feature_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().mean() < 0.5:
            continue
        if vals.nunique(dropna=True) <= 1:
            continue
        final_features.append(c)

    feature_cols = final_features

    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable morphology features: {feature_cols}")

    print(f"Events used: {len(df)}")
    print(f"Features used: {len(feature_cols)}")
    for c in feature_cols:
        print(" -", c)

    X_prepared, X_scaled = prepare_feature_matrix(df, feature_cols)

    n_components = min(8, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    # Cluster in PCA space.
    cluster_space = X_pca[:, :min(5, X_pca.shape[1])]

    if args.force_k is not None:
        k = args.force_k
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=0, n_init=30)
        model.fit(cluster_space)
        labels = model.predict(cluster_space)
        model_selection = pd.DataFrame([{
            "k": k,
            "bic": model.bic(cluster_space),
            "silhouette": silhouette_score(cluster_space, labels) if len(np.unique(labels)) > 1 else np.nan,
            "forced": True,
        }])
    else:
        min_cluster_size = max(3, int(round(0.05 * len(df))))
        max_k = min(args.max_k, max(args.min_k, len(df) // min_cluster_size))
        best, model_selection = choose_k_by_bic_with_min_size(
            cluster_space,
            min_k=args.min_k,
            max_k=max_k,
            min_cluster_size=min_cluster_size,
        )
        model = best["model"]
        labels = best["labels"]
        k = best["k"]

    probs = model.predict_proba(cluster_space)

    df["emg_morphology_cluster"] = labels
    df["emg_morphology_cluster_probability"] = probs.max(axis=1)

    df, cluster_summary = describe_clusters(df, feature_cols)

    # Save outputs.
    df.to_csv(out_dir / "refined_emg_morphology_features_with_clusters.csv", index=False)
    cluster_summary.to_csv(out_dir / "refined_emg_morphology_cluster_summary.csv", index=False)
    model_selection.to_csv(out_dir / "refined_gmm_model_selection.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "refined_features_used.csv", index=False)

    # PCA explained variance and loadings.
    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    explained.to_csv(out_dir / "refined_pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    loadings.to_csv(out_dir / "refined_pca_loadings.csv")

    # Composition tables.
    for cols, name in [
        (["group", "week", "emg_morphology_cluster", "suggested_morphology_label"], "refined_cluster_composition_by_group_week.csv"),
        (["primary_category", "emg_morphology_cluster", "suggested_morphology_label"], "refined_cluster_composition_by_category.csv"),
        (["qc_status", "emg_morphology_cluster", "suggested_morphology_label"], "refined_cluster_composition_by_qc_status.csv"),
    ]:
        valid = [c for c in cols if c in df.columns]
        if len(valid) >= 2:
            df.groupby(valid, as_index=False).size().to_csv(out_dir / name, index=False)

    # Plots.
    plot_pca(
        df,
        "emg_morphology_cluster",
        out_dir / "refined_pca_by_cluster.png",
        f"Refined EMG morphology PCA, GMM k={k}",
        label_outliers=True,
    )

    plot_pca_zoom(df, out_dir / "refined_pca_by_cluster_zoom.png")

    for color_col in ["group", "week", "primary_category", "qc_status", "suggested_morphology_label"]:
        if color_col in df.columns:
            plot_pca(
                df,
                color_col,
                out_dir / f"refined_pca_by_{color_col}.png",
                f"Refined PCA colored by {color_col}",
            )

    plot_feature_heatmap(
        df,
        feature_cols,
        out_dir / "refined_cluster_feature_heatmap.png",
    )

    print("\nDone.")
    print("Events used:", len(df))
    print("Chosen k:", k)
    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))
    print("\nSaved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
