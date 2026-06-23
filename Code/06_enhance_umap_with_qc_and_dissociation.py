from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "KjærbyLab/Project_PD_RBD_Katia/Data/prepared_data/manifests"
)

UMAP_IN = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "umap"
    / "emg_morphology_umap_embedding.csv"
)

QC_PATH = (
    BASE
    / "EMG_burst_detection_NREM_baseline"
    / "qc_ready"
    / "interactive_QC_annotations.csv"
)

OUT_DIR = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "umap"
    / "qc_dissociation_enhanced"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_state(x):
    x = str(x).strip()
    mapping = {
        "Wake": "Awake",
        "WK": "Awake",
        "W": "Awake",
        "wake": "Awake",
        "AWAKE": "Awake",
        "Awake": "Awake",
        "SWS": "NREM",
        "NREM": "NREM",
        "Nrem": "NREM",
        "PS": "REM",
        "REM": "REM",
        "Rem": "REM",
        "TR": "Undefined",
        "ND": "Undefined",
        "Undefined": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
        "": "Undefined",
    }
    return mapping.get(x, x)


def merge_qc(df):
    out = df.copy()

    if not QC_PATH.exists():
        out["qc_status"] = "not_reviewed"
        out["qc_notes"] = ""
        return out

    qc = pd.read_csv(QC_PATH)

    if "stable_event_key" in out.columns and "stable_event_key" in qc.columns:
        keep = [c for c in ["stable_event_key", "qc_status", "qc_notes"] if c in qc.columns]
        qc = qc[keep].drop_duplicates("stable_event_key", keep="last")
        out = out.merge(qc, on="stable_event_key", how="left")

    elif "qc_event_id" in out.columns and "qc_event_id" in qc.columns:
        keep = [c for c in ["qc_event_id", "qc_status", "qc_notes"] if c in qc.columns]
        qc = qc[keep].drop_duplicates("qc_event_id", keep="last")
        out = out.merge(qc, on="qc_event_id", how="left")

    else:
        out["qc_status"] = "not_reviewed"
        out["qc_notes"] = ""

    out["qc_status"] = out["qc_status"].fillna("not_reviewed")
    out["qc_notes"] = out["qc_notes"].fillna("")

    return out


def add_dissociation(df):
    out = df.copy()

    if "delta_REM" in out.columns:
        out["rem_wake_dissociation_recomputed"] = pd.to_numeric(
            out["delta_REM"],
            errors="coerce",
        )
    else:
        out["rem_wake_dissociation_recomputed"] = (
            pd.to_numeric(out.get("P_REM_EEGonly", np.nan), errors="coerce")
            - pd.to_numeric(out.get("P_REM_FULL", np.nan), errors="coerce")
        )

    out["rem_wake_dissociation_pos_recomputed"] = (
        out["rem_wake_dissociation_recomputed"].clip(lower=0)
    )

    if "max_EMG_z_existing" in out.columns:
        emg_z = pd.to_numeric(out["max_EMG_z_existing"], errors="coerce")
    elif "max_EMG_z" in out.columns:
        emg_z = pd.to_numeric(out["max_EMG_z"], errors="coerce")
    elif "emg_rms_z_max" in out.columns:
        emg_z = pd.to_numeric(out["emg_rms_z_max"], errors="coerce")
    else:
        emg_z = pd.Series(np.nan, index=out.index)

    out["emg_z_for_dissociation"] = emg_z

    out["rbd_dissociation_score_recomputed"] = (
        out["rem_wake_dissociation_pos_recomputed"]
        * out["emg_z_for_dissociation"].clip(lower=0)
    )

    out["candidate_rbd_like_dissociation_recomputed"] = (
        (pd.to_numeric(out.get("P_REM_EEGonly", np.nan), errors="coerce") >= 0.60)
        & (out["rem_wake_dissociation_recomputed"] >= 0.30)
        & (out["emg_z_for_dissociation"] >= 2.0)
    ).fillna(False)

    return out


def add_rem_flags(df):
    out = df.copy()

    rem_relevant = [
        "stable_REM_EMG_burst",
        "EMG_suppressed_REM",
        "mixed_REM_Wake_transition",
    ]

    out["REM_relevant_auto"] = out["primary_category"].isin(rem_relevant)

    if "EEGonly_state_center" in out.columns:
        out["EEGonly_center_REM"] = out["EEGonly_state_center"].map(normalize_state).eq("REM")
    else:
        out["EEGonly_center_REM"] = False

    if "P_REM_EEGonly" in out.columns:
        out["EEGonly_PREM_ge_060"] = pd.to_numeric(
            out["P_REM_EEGonly"],
            errors="coerce",
        ).ge(0.60)
    else:
        out["EEGonly_PREM_ge_060"] = False

    out["REM_like_union_recomputed"] = (
        out["REM_relevant_auto"]
        | out["EEGonly_center_REM"]
        | out["EEGonly_PREM_ge_060"]
        | out["candidate_rbd_like_dissociation_recomputed"]
    )

    return out


def plot_umap_categorical(df, color_col, out_path, title):
    fig, ax = plt.subplots(figsize=(8.5, 6.4))

    vals = df[color_col].fillna("NA").astype(str)
    cats = sorted(vals.unique())

    for cat in cats:
        sub = df[vals == cat]
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=8,
            alpha=0.70,
            label=cat,
        )

    if len(cats) <= 15:
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
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_umap_highlight(df, mask, out_path, title, label):
    fig, ax = plt.subplots(figsize=(8.5, 6.4))

    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)

    ax.scatter(
        df.loc[~mask, "umap1"],
        df.loc[~mask, "umap2"],
        s=5,
        alpha=0.17,
        label="Other",
    )

    ax.scatter(
        df.loc[mask, "umap1"],
        df.loc[mask, "umap2"],
        s=18,
        alpha=0.90,
        label=label,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_crosstab(df, row_col, col_col, filename_prefix):
    count_tab = pd.crosstab(df[row_col], df[col_col], dropna=False)
    pct_tab = pd.crosstab(df[row_col], df[col_col], normalize="index", dropna=False) * 100

    count_tab.to_csv(OUT_DIR / f"{filename_prefix}_counts.csv")
    pct_tab.to_csv(OUT_DIR / f"{filename_prefix}_row_percent.csv")

    return count_tab, pct_tab


def main():
    df = pd.read_csv(UMAP_IN)

    df = merge_qc(df)
    df = add_dissociation(df)
    df = add_rem_flags(df)

    out_csv = OUT_DIR / "emg_morphology_umap_embedding_with_qc_and_dissociation.csv"
    df.to_csv(out_csv, index=False)

    print("Events:", len(df))
    print("QC reviewed:", int((df["qc_status"] != "not_reviewed").sum()))
    print(
        "Dissociation-positive:",
        int(df["candidate_rbd_like_dissociation_recomputed"].sum()),
    )
    print("REM-relevant auto:", int(df["REM_relevant_auto"].sum()))
    print("REM-like union:", int(df["REM_like_union_recomputed"].sum()))

    # Main categorical plots.
    for col in [
        "qc_status",
        "primary_category",
        "emg_morphology_broad_label",
        "emg_morphology_fine_label",
        "emg_morphology_cluster",
        "group",
        "week",
    ]:
        if col in df.columns:
            plot_umap_categorical(
                df,
                col,
                OUT_DIR / f"umap_by_{col}.png",
                f"UMAP colored by {col}",
            )

    # QC-specific highlights.
    for status in [
        "possible_RBD_like",
        "transition_event",
        "artifact",
        "exclude",
        "real_burst_but_wake",
        "uncertain",
        "not_reviewed",
    ]:
        mask = df["qc_status"].eq(status)
        if mask.any():
            plot_umap_highlight(
                df,
                mask,
                OUT_DIR / f"umap_highlight_qc_{status}.png",
                f"UMAP highlight: QC {status}",
                status,
            )

    # REM / dissociation highlights.
    highlight_defs = {
        "REM_relevant_auto": df["REM_relevant_auto"],
        "EEGonly_center_REM": df["EEGonly_center_REM"],
        "EEGonly_PREM_ge_060": df["EEGonly_PREM_ge_060"],
        "candidate_rbd_like_dissociation_recomputed": df["candidate_rbd_like_dissociation_recomputed"],
        "REM_like_union_recomputed": df["REM_like_union_recomputed"],
    }

    for name, mask in highlight_defs.items():
        plot_umap_highlight(
            df,
            mask,
            OUT_DIR / f"umap_highlight_{name}.png",
            f"UMAP highlight: {name}",
            name,
        )

    # Summary tables.
    if "emg_morphology_broad_label" in df.columns:
        save_crosstab(
            df,
            "qc_status",
            "emg_morphology_broad_label",
            "qc_status_by_broad_morphology",
        )

        save_crosstab(
            df,
            "primary_category",
            "emg_morphology_broad_label",
            "primary_category_by_broad_morphology",
        )

        save_crosstab(
            df,
            "REM_relevant_auto",
            "emg_morphology_broad_label",
            "REM_relevant_auto_by_broad_morphology",
        )

        save_crosstab(
            df,
            "candidate_rbd_like_dissociation_recomputed",
            "emg_morphology_broad_label",
            "dissociation_candidate_by_broad_morphology",
        )

    if "group" in df.columns and "week" in df.columns:
        group_week = (
            df.groupby(["group", "week", "emg_morphology_broad_label"], as_index=False)
            .agg(
                n_events=("qc_event_id", "count"),
                n_REM_relevant=("REM_relevant_auto", "sum"),
                n_dissociation_candidate=("candidate_rbd_like_dissociation_recomputed", "sum"),
                mean_rbd_dissociation_score=("rbd_dissociation_score_recomputed", "mean"),
                median_rbd_dissociation_score=("rbd_dissociation_score_recomputed", "median"),
            )
        )

        group_week["total_events_group_week"] = group_week.groupby(["group", "week"])["n_events"].transform("sum")
        group_week["percent_of_events"] = 100 * group_week["n_events"] / group_week["total_events_group_week"]

        group_week.to_csv(
            OUT_DIR / "group_week_by_broad_morphology_with_dissociation.csv",
            index=False,
        )

    print("\nSaved enhanced UMAP outputs to:")
    print(OUT_DIR)
    print(out_csv)


if __name__ == "__main__":
    main()
