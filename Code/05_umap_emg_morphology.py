from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

try:
    import umap
except Exception as e:
    raise ImportError(
        "UMAP is not installed. Run: pip install umap-learn"
    ) from e


BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "KjærbyLab/Project_PD_RBD_Katia/Data/prepared_data/manifests"
)

DEFAULT_IN = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "refined_emg_morphology_features_with_metaclasses.csv"
)

DEFAULT_FALLBACK_IN = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "refined_emg_morphology_features_with_clusters.csv"
)

DEFAULT_FEATURES = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "refined_features_used.csv"
)

DEFAULT_OUT_DIR = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "umap"
)


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


def auto_add_metaclass_labels(df):
    out = df.copy()

    if "emg_morphology_cluster" not in out.columns:
        return out

    fine_map = {
        0: "ultra_brief_low_amplitude_or_mixed",
        1: "brief_phasic_twitch_like",
        2: "sustained_moderate_complex",
        3: "brief_high_duty_phasic",
        4: "clustered_phasic_high_background_tone",
        5: "sustained_high_amplitude_tonic_like",
    }

    broad_map = {
        0: "phasic_or_uncertain",
        1: "phasic",
        2: "sustained_tonic",
        3: "phasic",
        4: "clustered_phasic_high_tone",
        5: "sustained_tonic",
    }

    cluster_numeric = pd.to_numeric(out["emg_morphology_cluster"], errors="coerce").astype("Int64")

    if "emg_morphology_fine_label" not in out.columns:
        out["emg_morphology_fine_label"] = cluster_numeric.map(fine_map)

    if "emg_morphology_broad_label" not in out.columns:
        out["emg_morphology_broad_label"] = cluster_numeric.map(broad_map)

    return out


def winsorize_series(x, lo=0.01, hi=0.99):
    x = pd.to_numeric(x, errors="coerce")

    if x.notna().sum() < 5:
        return x

    qlo = x.quantile(lo)
    qhi = x.quantile(hi)

    return x.clip(qlo, qhi)


def prepare_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()

    if "active_fraction_in_event" in X.columns:
        X["active_fraction_in_event"] = pd.to_numeric(
            X["active_fraction_in_event"],
            errors="coerce",
        ).clip(lower=0, upper=1)

    for c in LOG1P_FEATURES:
        if c in X.columns:
            vals = pd.to_numeric(X[c], errors="coerce").clip(lower=0)
            X[c] = np.log1p(vals)

    for c in X.columns:
        X[c] = winsorize_series(X[c])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled


def add_rem_flags(df):
    out = df.copy()

    if "primary_category" not in out.columns:
        out["primary_category"] = ""

    rem_relevant = [
        "stable_REM_EMG_burst",
        "EMG_suppressed_REM",
        "mixed_REM_Wake_transition",
    ]

    out["umap_REM_relevant_auto"] = out["primary_category"].isin(rem_relevant)

    if "EEGonly_state_center" in out.columns:
        out["umap_EEGonly_center_REM"] = out["EEGonly_state_center"].astype(str).str.upper().eq("REM")
    else:
        out["umap_EEGonly_center_REM"] = False

    if "P_REM_EEGonly" in out.columns:
        out["umap_EEGonly_PREM_ge_060"] = pd.to_numeric(
            out["P_REM_EEGonly"],
            errors="coerce",
        ).ge(0.60)
    else:
        out["umap_EEGonly_PREM_ge_060"] = False

    if "candidate_rbd_like_dissociation" in out.columns:
        out["umap_dissociation_positive"] = out["candidate_rbd_like_dissociation"].fillna(False).astype(bool)
    elif "rbd_dissociation_score" in out.columns:
        out["umap_dissociation_positive"] = pd.to_numeric(
            out["rbd_dissociation_score"],
            errors="coerce",
        ).gt(0)
    else:
        out["umap_dissociation_positive"] = False

    out["umap_REM_like_union"] = (
        out["umap_REM_relevant_auto"]
        | out["umap_EEGonly_center_REM"]
        | out["umap_EEGonly_PREM_ge_060"]
        | out["umap_dissociation_positive"]
    )

    return out


def plot_umap(df, color_col, out_path, title, max_categories=14):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    if color_col not in df.columns:
        ax.scatter(df["umap1"], df["umap2"], s=8, alpha=0.65)
    else:
        vals = df[color_col].fillna("NA").astype(str)
        cats = sorted(vals.unique())

        if len(cats) > max_categories:
            # For too many categories, plot as gray.
            ax.scatter(df["umap1"], df["umap2"], s=8, alpha=0.65)
        else:
            for cat in cats:
                sub = df[vals == cat]
                ax.scatter(
                    sub["umap1"],
                    sub["umap2"],
                    s=8,
                    alpha=0.70,
                    label=cat,
                )

            ax.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
                markerscale=2,
            )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_umap_binary_highlight(df, flag_col, out_path, title):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    flag = df[flag_col].fillna(False).astype(bool)

    ax.scatter(
        df.loc[~flag, "umap1"],
        df.loc[~flag, "umap2"],
        s=6,
        alpha=0.20,
        label="Other",
    )

    ax.scatter(
        df.loc[flag, "umap1"],
        df.loc[flag, "umap2"],
        s=14,
        alpha=0.85,
        label=flag_col,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_IN))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument(
        "--rem-like-only",
        action="store_true",
        help="Fit UMAP only on REM-like union events.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        input_path = DEFAULT_FALLBACK_IN

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = auto_add_metaclass_labels(df)
    df = add_rem_flags(df)

    if args.rem_like_only:
        df = df[df["umap_REM_like_union"]].copy()
        out_dir = out_dir / "rem_like_only"
        out_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(args.features)
    feature_cols = features_df["feature"].tolist()
    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable features: {feature_cols}")

    print(f"Events used: {len(df)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")

    X_scaled = prepare_feature_matrix(df, feature_cols)

    n_pca = min(args.pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)

    pca = PCA(n_components=n_pca, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )

    embedding = reducer.fit_transform(X_pca)

    df["umap1"] = embedding[:, 0]
    df["umap2"] = embedding[:, 1]

    out_csv = out_dir / "emg_morphology_umap_embedding.csv"
    df.to_csv(out_csv, index=False)

    pd.DataFrame({
        "parameter": [
            "input",
            "n_events",
            "n_neighbors",
            "min_dist",
            "metric",
            "random_state",
            "pca_components",
            "rem_like_only",
        ],
        "value": [
            str(input_path),
            len(df),
            args.n_neighbors,
            args.min_dist,
            args.metric,
            args.random_state,
            n_pca,
            args.rem_like_only,
        ],
    }).to_csv(out_dir / "umap_parameters.csv", index=False)

    # Main plots.
    for col in [
        "emg_morphology_cluster",
        "emg_morphology_broad_label",
        "emg_morphology_fine_label",
        "primary_category",
        "qc_status",
        "group",
        "week",
    ]:
        if col in df.columns:
            plot_umap(
                df,
                col,
                out_dir / f"umap_by_{col}.png",
                f"EMG morphology UMAP colored by {col}",
            )

    # REM/RBD-focused overlays.
    for flag_col in [
        "umap_REM_relevant_auto",
        "umap_EEGonly_center_REM",
        "umap_EEGonly_PREM_ge_060",
        "umap_dissociation_positive",
        "umap_REM_like_union",
    ]:
        if flag_col in df.columns:
            plot_umap_binary_highlight(
                df,
                flag_col,
                out_dir / f"umap_highlight_{flag_col}.png",
                f"UMAP highlight: {flag_col}",
            )

    print("\nSaved:")
    print(out_csv)
    print(out_dir)


if __name__ == "__main__":
    main()
