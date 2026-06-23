from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from supervisor_conclusion_layer import render_conclusion_layer
except Exception:
    render_conclusion_layer = None



# =============================================================================
# PATHS
# =============================================================================

BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "PD-Katia/Data/prepared_data/manifests"
)

DEFAULT_ENHANCED_UMAP = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "umap"
    / "qc_dissociation_enhanced"
    / "emg_morphology_umap_embedding_with_qc_and_dissociation.csv"
)

DEFAULT_QC = (
    BASE
    / "EMG_burst_detection_NREM_baseline"
    / "qc_ready"
    / "interactive_QC_annotations.csv"
)

DEFAULT_OUT = (
    BASE
    / "EMG_unsupervised_morphology"
    / "supervisor_plots"
)


# =============================================================================
# CLUSTER NAMES AND FEATURE DICTIONARY
# =============================================================================

CLUSTER_NAME_MAP = {
    0: "Micro / low-amplitude phasic events",
    1: "Brief twitch-like phasic events",
    2: "Sustained complex activation",
    3: "Compact high-duty phasic bursts",
    4: "Clustered bursts on high tone",
    5: "Large sustained tonic-like activation",
}

CLUSTER_SHORT_MAP = {
    0: "micro-phasic",
    1: "brief phasic",
    2: "sustained complex",
    3: "compact phasic",
    4: "clustered high-tone",
    5: "large sustained tonic",
}

CLUSTER_BROAD_MAP = {
    0: "phasic / uncertain",
    1: "phasic",
    2: "sustained / tonic",
    3: "phasic",
    4: "clustered + high tone",
    5: "sustained / tonic",
}

CLUSTER_EXPLANATION_MAP = {
    0: (
        "Very short or low-amplitude events. These may include tiny twitches, "
        "low-confidence bursts, or ambiguous EMG fluctuations."
    ),
    1: (
        "Short twitch-like events, usually compact in time and consistent with "
        "brief phasic motor activation."
    ),
    2: (
        "Longer, more complex events with sustained activation and multiple components. "
        "This class may capture intermediate sustained motor activity."
    ),
    3: (
        "Short but high-duty events, meaning a large fraction of the short event is "
        "occupied by active EMG. These look like compact phasic bursts."
    ),
    4: (
        "Repeated bursts occurring on elevated background EMG tone. This class may "
        "reflect clustered twitching, unstable tone, or transition-like motor activity."
    ),
    5: (
        "Large, longer-lasting, high-amplitude events. This is the class most consistent "
        "with sustained tonic/tumbling-like motor activation."
    ),
}

FEATURE_PRETTY = {
    "morph_duration_sec": "Duration",
    "emg_rms_z_mean": "Mean EMG amplitude",
    "emg_rms_z_max": "Max EMG amplitude",
    "emg_rms_z_p95": "95th pct EMG amplitude",
    "emg_rms_auc": "Total EMG activity",
    "emg_event_to_background_rms_ratio": "Event/background contrast",
    "background_rms_median": "Background tone",
    "background_rms_scale": "Background variability",
    "pre_event_rms_median": "Pre-event tone",
    "post_event_rms_median": "Post-event tone",
    "pre_post_tone_change": "Tone change after event",
    "n_subbursts": "Number of sub-bursts",
    "subburst_rate_per_sec": "Sub-burst rate",
    "total_active_duration_sec": "Total active EMG time",
    "active_fraction_in_event": "Active fraction",
    "mean_subburst_duration_sec": "Mean sub-burst duration",
    "max_subburst_duration_sec": "Max sub-burst duration",
    "n_rms_peaks": "Number of EMG peaks",
    "rms_peak_rate_per_sec": "Peak rate",
    "rms_peak_interval_cv": "Peak irregularity",
    "rms_peak_regularity_score": "Peak regularity",
    "rms_peak_to_mean_ratio": "Peakiness",
    "rms_skewness": "Shape asymmetry",
    "rms_kurtosis": "Sharpness / outlier peaks",
    "emg_spectral_centroid_hz": "Spectral centroid",
    "emg_spectral_median_hz": "Median EMG frequency",
    "emg_spectral_high_fraction_50_150": "High-frequency EMG fraction",
}

FEATURE_HELP = {
    "Duration": "How long the event lasts.",
    "Mean EMG amplitude": "Average EMG strength during the event.",
    "Max EMG amplitude": "Largest EMG activation during the event.",
    "Total EMG activity": "Overall amount of EMG activity across the event.",
    "Background tone": "EMG tone around the event, outside the main burst.",
    "Number of sub-bursts": "How many smaller bursts are contained inside the event.",
    "Active fraction": "How much of the event is occupied by active EMG.",
    "Peakiness": "Whether the event is sharp/peaky or broad/sustained.",
    "Peak irregularity": "How irregular the intervals between peaks are.",
    "Peak regularity": "How regular repeated peaks are.",
    "High-frequency EMG fraction": "How much EMG power lies in a high-frequency range.",
}


# =============================================================================
# DATA LOADING / CLEANING
# =============================================================================

def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_table(path):
    return pd.read_csv(path)


def add_cluster_labels(df):
    out = df.copy()

    if "emg_morphology_cluster" not in out.columns:
        return out

    cluster = pd.to_numeric(out["emg_morphology_cluster"], errors="coerce").astype("Int64")

    out["cluster_name"] = cluster.map(CLUSTER_NAME_MAP)
    out["cluster_short_name"] = cluster.map(CLUSTER_SHORT_MAP)
    out["broad_morphology_name"] = cluster.map(CLUSTER_BROAD_MAP)
    out["cluster_explanation"] = cluster.map(CLUSTER_EXPLANATION_MAP)

    return out


def merge_qc(df, qc_path):
    out = df.copy()
    qc_path = Path(qc_path)

    # Remove stale/suffixed QC columns.
    for c in ["qc_status", "qc_notes", "qc_status_x", "qc_status_y", "qc_notes_x", "qc_notes_y"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    if not qc_path.exists():
        out["qc_status"] = "not_reviewed"
        out["qc_notes"] = ""
        return out

    qc = pd.read_csv(qc_path)

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


def add_rem_and_dissociation_flags(df):
    out = df.copy()

    if "primary_category" not in out.columns:
        out["primary_category"] = ""

    rem_relevant_categories = [
        "stable_REM_EMG_burst",
        "EMG_suppressed_REM",
        "mixed_REM_Wake_transition",
    ]

    out["REM_relevant_auto"] = out["primary_category"].isin(rem_relevant_categories)

    if "EEGonly_state_center" in out.columns:
        out["EEGonly_center_REM"] = out["EEGonly_state_center"].astype(str).str.upper().eq("REM")
    else:
        out["EEGonly_center_REM"] = False

    if "P_REM_EEGonly" in out.columns:
        out["EEGonly_PREM_ge_060"] = pd.to_numeric(out["P_REM_EEGonly"], errors="coerce").ge(0.60)
    else:
        out["EEGonly_PREM_ge_060"] = False

    # Recompute dissociation robustly.
    if "delta_REM" in out.columns:
        out["rem_wake_dissociation_recomputed"] = pd.to_numeric(out["delta_REM"], errors="coerce")
    elif "P_REM_EEGonly" in out.columns and "P_REM_FULL" in out.columns:
        out["rem_wake_dissociation_recomputed"] = (
            pd.to_numeric(out["P_REM_EEGonly"], errors="coerce")
            - pd.to_numeric(out["P_REM_FULL"], errors="coerce")
        )
    else:
        out["rem_wake_dissociation_recomputed"] = np.nan

    emg_col = first_existing_col(out, ["max_EMG_z_existing", "max_EMG_z", "emg_rms_z_max"])
    if emg_col is not None:
        out["emg_z_for_dissociation"] = pd.to_numeric(out[emg_col], errors="coerce")
    else:
        out["emg_z_for_dissociation"] = np.nan

    out["rbd_dissociation_score_recomputed"] = (
        out["rem_wake_dissociation_recomputed"].clip(lower=0)
        * out["emg_z_for_dissociation"].clip(lower=0)
    )

    if "P_REM_EEGonly" in out.columns:
        prem = pd.to_numeric(out["P_REM_EEGonly"], errors="coerce")
    else:
        prem = pd.Series(np.nan, index=out.index)

    out["candidate_rbd_like_dissociation_recomputed"] = (
        (prem >= 0.60)
        & (out["rem_wake_dissociation_recomputed"] >= 0.30)
        & (out["emg_z_for_dissociation"] >= 2.0)
    ).fillna(False)

    out["REM_like_union"] = (
        out["REM_relevant_auto"]
        | out["EEGonly_center_REM"]
        | out["EEGonly_PREM_ge_060"]
        | out["candidate_rbd_like_dissociation_recomputed"]
    )

    return out


def get_group_week_label(df):
    out = df.copy()
    out["week_num"] = pd.to_numeric(out["week"], errors="coerce")
    out["week_int"] = out["week_num"].round().astype("Int64")
    out["group_week"] = out["group"].astype(str) + " W" + out["week_int"].astype(str)
    return out


def ordered_group_week_labels(df):
    df = get_group_week_label(df)
    order = []

    for group in ["WT", "PD"]:
        weeks = sorted(
            pd.to_numeric(df.loc[df["group"] == group, "week"], errors="coerce")
            .dropna()
            .unique()
        )
        for w in weeks:
            label = f"{group} W{int(w)}"
            if label not in order:
                order.append(label)

    existing = list(df["group_week"].dropna().unique())
    extras = sorted([x for x in existing if x not in order])

    return order + extras


def load_analysis_table(umap_path, qc_path):
    df = load_table(umap_path)
    df = merge_qc(df, qc_path)
    df = add_cluster_labels(df)
    df = add_rem_and_dissociation_flags(df)

    return df


# =============================================================================
# FILTERING
# =============================================================================

def apply_sidebar_filters(df):
    out = df.copy()

    st.sidebar.header("Display filters")

    if "group" in out.columns:
        groups = sorted(out["group"].dropna().unique())
        selected_groups = st.sidebar.multiselect("Group", groups, default=groups)
        out = out[out["group"].isin(selected_groups)].copy()

    if "week" in out.columns:
        weeks = sorted(out["week"].dropna().unique())
        selected_weeks = st.sidebar.multiselect("Week", weeks, default=weeks)
        out = out[out["week"].isin(selected_weeks)].copy()

    if "primary_category" in out.columns:
        cats = sorted(out["primary_category"].dropna().unique())
        default_cats = cats
        selected_cats = st.sidebar.multiselect("Automatic category", cats, default=default_cats)
        out = out[out["primary_category"].isin(selected_cats)].copy()

    subset_options = {
        "All events": pd.Series(True, index=out.index),
        "REM-relevant automatic": out.get("REM_relevant_auto", pd.Series(False, index=out.index)),
        "REM-like union": out.get("REM_like_union", pd.Series(False, index=out.index)),
        "Dissociation-positive candidates": out.get(
            "candidate_rbd_like_dissociation_recomputed",
            pd.Series(False, index=out.index),
        ),
        "QC reviewed only": out.get("qc_status", pd.Series("not_reviewed", index=out.index)).ne("not_reviewed"),
        "QC possible RBD-like": out.get("qc_status", pd.Series("", index=out.index)).eq("possible_RBD_like"),
    }

    subset_name = st.sidebar.selectbox(
        "Biological subset",
        list(subset_options.keys()),
        index=0,
        help="For the meeting, use All events for the atlas, then REM-relevant, Dissociation-positive, and QC possible RBD-like for biological interpretation.",
    )

    mask = subset_options[subset_name].fillna(False).astype(bool)
    out = out[mask].copy()

    if "broad_morphology_name" in out.columns:
        morphs = sorted(out["broad_morphology_name"].dropna().unique())
        selected_morphs = st.sidebar.multiselect("Broad morphology", morphs, default=morphs)
        out = out[out["broad_morphology_name"].isin(selected_morphs)].copy()

    return out, subset_name


# =============================================================================
# PLOTS
# =============================================================================

def plot_umap(df, color_col, highlight_kind="None"):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    if "umap1" not in df.columns or "umap2" not in df.columns:
        ax.text(0.5, 0.5, "No UMAP coordinates found", ha="center", va="center")
        return fig

    plot_df = df.copy()

    if highlight_kind != "None":
        if highlight_kind == "Dissociation-positive":
            mask = plot_df["candidate_rbd_like_dissociation_recomputed"].fillna(False).astype(bool)
            label = "Dissociation-positive"
        elif highlight_kind == "REM-relevant":
            mask = plot_df["REM_relevant_auto"].fillna(False).astype(bool)
            label = "REM-relevant"
        elif highlight_kind.startswith("QC: "):
            status = highlight_kind.replace("QC: ", "")
            mask = plot_df["qc_status"].eq(status)
            label = status
        else:
            mask = pd.Series(False, index=plot_df.index)
            label = highlight_kind

        ax.scatter(plot_df.loc[~mask, "umap1"], plot_df.loc[~mask, "umap2"], s=7, alpha=0.15, label="Other")
        ax.scatter(plot_df.loc[mask, "umap1"], plot_df.loc[mask, "umap2"], s=18, alpha=0.85, label=label)
        ax.legend(loc="best", fontsize=8)
    else:
        vals = plot_df[color_col].fillna("NA").astype(str)
        cats = sorted(vals.unique())

        for cat in cats:
            sub = plot_df[vals == cat]
            ax.scatter(sub["umap1"], sub["umap2"], s=8, alpha=0.70, label=cat)

        if len(cats) <= 16:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("EMG morphology map")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def composition_matrix(df, category_col, unit="mouse-level"):
    if df.empty or category_col not in df.columns:
        return pd.DataFrame()

    df = get_group_week_label(df)

    if unit == "event-level" or "mouse_id" not in df.columns:
        comp = pd.crosstab(df["group_week"], df[category_col], normalize="index") * 100
        return comp

    counts = (
        df.groupby(["group_week", "mouse_id", category_col], dropna=False)
        .size()
        .reset_index(name="n")
    )

    totals = counts.groupby(["group_week", "mouse_id"])["n"].transform("sum")
    counts["pct"] = 100 * counts["n"] / totals.replace(0, np.nan)

    mouse_tab = counts.pivot_table(
        index=["group_week", "mouse_id"],
        columns=category_col,
        values="pct",
        fill_value=0,
    )

    group_tab = mouse_tab.groupby("group_week").mean()

    return group_tab


def plot_stacked_composition(comp, title, order=None):
    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    if comp.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    if order is not None:
        comp = comp.reindex([x for x in order if x in comp.index])

    comp.plot(kind="bar", stacked=True, ax=ax, width=0.78)

    ax.set_ylabel("% of events")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def contrast_heatmap(comp):
    if comp.empty:
        return pd.DataFrame()

    labels = list(comp.index)

    def find_label(group, when):
        candidates = []
        for x in labels:
            if not str(x).startswith(group + " W"):
                continue
            try:
                week = int(str(x).split("W")[-1])
                candidates.append((week, x))
            except Exception:
                pass

        if not candidates:
            return None

        candidates = sorted(candidates)
        if when == "early":
            return candidates[0][1]
        if when == "late":
            return candidates[-1][1]
        return None

    wt_early = find_label("WT", "early")
    wt_late = find_label("WT", "late")
    pd_early = find_label("PD", "early")
    pd_late = find_label("PD", "late")

    rows = {}

    def add(name, a, b):
        if a in comp.index and b in comp.index:
            rows[name] = comp.loc[a] - comp.loc[b]

    add("PD late - WT late", pd_late, wt_late)
    add("PD early - WT early", pd_early, wt_early)
    add("PD late - PD early", pd_late, pd_early)
    add("WT late - WT early", wt_late, wt_early)

    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def plot_difference_heatmap(diff, title):
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    if diff.empty:
        ax.text(0.5, 0.5, "No valid contrasts", ha="center", va="center")
        return fig

    max_abs = np.nanmax(np.abs(diff.values))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    im = ax.imshow(diff.values, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

    ax.set_yticks(np.arange(diff.shape[0]))
    ax.set_yticklabels(diff.index)

    ax.set_xticks(np.arange(diff.shape[1]))
    ax.set_xticklabels(diff.columns, rotation=35, ha="right")

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            val = diff.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Difference in percentage points")

    fig.tight_layout()
    return fig


def plot_morphology_by_subsets(df, morphology_col):
    rows = []

    subset_defs = {
        "All events": pd.Series(True, index=df.index),
        "REM-relevant": df["REM_relevant_auto"].fillna(False).astype(bool),
        "Dissociation-positive": df["candidate_rbd_like_dissociation_recomputed"].fillna(False).astype(bool),
        "QC possible RBD-like": df["qc_status"].eq("possible_RBD_like"),
    }

    for label, mask in subset_defs.items():
        sub = df[mask].copy()
        if len(sub) == 0:
            continue

        counts = sub[morphology_col].value_counts()
        total = counts.sum()

        for morph, n in counts.items():
            rows.append({
                "subset": label,
                "morphology": morph,
                "percent": 100 * n / total,
                "n": int(n),
                "total": int(total),
            })

    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    if tab.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, tab

    comp = tab.pivot_table(index="subset", columns="morphology", values="percent", fill_value=0)
    order = [x for x in ["All events", "REM-relevant", "Dissociation-positive", "QC possible RBD-like"] if x in comp.index]
    comp = comp.reindex(order)

    comp.plot(kind="bar", stacked=True, ax=ax, width=0.75)

    ax.set_ylabel("% of events")
    ax.set_title("Morphology distribution across biologically relevant subsets")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig, tab


def plot_qc_by_morphology(df, morphology_col):
    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    reviewed = df[df["qc_status"].ne("not_reviewed")].copy()

    if reviewed.empty:
        ax.text(0.5, 0.5, "No reviewed QC events in current subset", ha="center", va="center")
        return fig, pd.DataFrame()

    comp = pd.crosstab(reviewed[morphology_col], reviewed["qc_status"], normalize="index") * 100
    comp.plot(kind="bar", stacked=True, ax=ax, width=0.75)

    ax.set_ylabel("% within morphology class")
    ax.set_title("QC outcome by morphology class, reviewed events only")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    count_tab = pd.crosstab(reviewed[morphology_col], reviewed["qc_status"], dropna=False)
    count_tab["reviewed_total"] = count_tab.sum(axis=1)

    if "possible_RBD_like" in count_tab.columns:
        count_tab["possible_RBD_like_yield_%"] = (
            100 * count_tab["possible_RBD_like"] / count_tab["reviewed_total"].replace(0, np.nan)
        )

    return fig, count_tab


def plot_mouse_level_dot(df, morphology_col, selected_morphology):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    if "mouse_id" not in df.columns:
        ax.text(0.5, 0.5, "No mouse_id column", ha="center", va="center")
        return fig

    df = get_group_week_label(df)

    tmp = (
        df.groupby(["group_week", "mouse_id"])
        .apply(lambda x: 100 * (x[morphology_col] == selected_morphology).mean())
        .reset_index(name="percent")
    )

    order = ordered_group_week_labels(df)
    rng = np.random.default_rng(4)
    means = []

    for i, gw in enumerate(order):
        sub = tmp[tmp["group_week"] == gw]
        vals = sub["percent"].to_numpy(dtype=float)

        means.append(np.nanmean(vals) if len(vals) else np.nan)

        jitter = rng.normal(0, 0.045, size=len(sub))
        xs = np.full(len(sub), i) + jitter

        ax.scatter(xs, vals, s=48, zorder=3)

        for x, y, mid in zip(xs, vals, sub["mouse_id"]):
            ax.text(x + 0.035, y, str(int(mid)), fontsize=8, va="center")

    ax.bar(range(len(order)), means, alpha=0.35, zorder=1)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(f"% {selected_morphology}")
    ax.set_title(f"Mouse-level burden: {selected_morphology}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def feature_z_profiles(df, label_col):
    feature_cols = [c for c in FEATURE_PRETTY if c in df.columns]

    if not feature_cols or label_col not in df.columns:
        return pd.DataFrame()

    med = df.groupby(label_col)[feature_cols].median(numeric_only=True)
    global_med = df[feature_cols].median(numeric_only=True)
    global_std = df[feature_cols].std(numeric_only=True).replace(0, np.nan)

    z = (med - global_med) / global_std
    z = z.rename(columns=FEATURE_PRETTY)

    return z


def plot_feature_driver_bars(z, selected_label, top_n=6):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    if z.empty or selected_label not in z.index:
        ax.text(0.5, 0.5, "No feature profile available", ha="center", va="center")
        return fig

    s = z.loc[selected_label].dropna()

    selected = pd.concat([
        s.sort_values(ascending=False).head(top_n),
        s.sort_values(ascending=True).head(top_n),
    ]).drop_duplicates()

    selected = selected.sort_values()

    ax.barh(selected.index, selected.values)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Feature level relative to all events, z-score")
    ax.set_title(f"What defines: {selected_label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def cluster_summary_cards(df):
    rows = []

    for cluster_id in sorted(pd.to_numeric(df["emg_morphology_cluster"], errors="coerce").dropna().unique()):
        cluster_id = int(cluster_id)
        name = CLUSTER_NAME_MAP.get(cluster_id, f"Cluster {cluster_id}")

        sub = df[pd.to_numeric(df["emg_morphology_cluster"], errors="coerce") == cluster_id]

        rows.append({
            "cluster": cluster_id,
            "name": name,
            "broad_class": CLUSTER_BROAD_MAP.get(cluster_id, ""),
            "n_events": len(sub),
            "plain_explanation": CLUSTER_EXPLANATION_MAP.get(cluster_id, ""),
        })

    return pd.DataFrame(rows)


def save_current_figure(fig, filename):
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    out_path = DEFAULT_OUT / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    st.caption(f"Saved: {out_path}")


# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(page_title="Supervisor EMG morphology results", layout="wide")

st.title("Supervisor view: EMG morphology, REM dissociation, and RBD-like events")

st.markdown(
    """
    This is a simplified results app for the meeting. It only shows the figures needed to explain
    the analysis after EMG burst detection: morphology features, unsupervised classes, QC validation,
    REM/dissociation enrichment, and preliminary PD/WT week comparisons.
    """
)

st.sidebar.header("Input files")

umap_path = st.sidebar.text_input(
    "Enhanced UMAP / morphology table",
    str(DEFAULT_ENHANCED_UMAP),
)

qc_path = st.sidebar.text_input(
    "QC annotations",
    str(DEFAULT_QC),
)

if not Path(umap_path).exists():
    st.error(f"Could not find enhanced UMAP table:\n{umap_path}")
    st.stop()

df_all = load_analysis_table(umap_path, qc_path)

if df_all.empty:
    st.error("Loaded table is empty.")
    st.stop()

df, subset_name = apply_sidebar_filters(df_all)

if df.empty:
    st.warning("No events after filters.")
    st.stop()

morphology_col = st.sidebar.selectbox(
    "Morphology label level",
    ["broad_morphology_name", "cluster_short_name", "cluster_name"],
    index=0,
)

# -------------------------------------------------------------------------
# Top summary
# -------------------------------------------------------------------------

st.subheader("Meeting summary")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Events shown", len(df))
c2.metric("Mice", df["mouse_id"].nunique() if "mouse_id" in df.columns else "NA")
c3.metric("QC reviewed", int(df["qc_status"].ne("not_reviewed").sum()))
c4.metric("QC possible RBD-like", int(df["qc_status"].eq("possible_RBD_like").sum()))
c5.metric("REM-relevant", int(df["REM_relevant_auto"].sum()))
c6.metric("Dissociation+", int(df["candidate_rbd_like_dissociation_recomputed"].sum()))

st.info(
    """
    Main message: the EMG events are not homogeneous. The unsupervised analysis organizes them
    into interpretable morphology classes ranging from brief/phasic events to clustered high-tone
    events and sustained/tonic-like activation. The key question is whether REM-relevant,
    dissociation-positive, and QC possible RBD-like events are enriched in particular morphologies.
    """
)

# -------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------


if render_conclusion_layer is not None:
    try:
        render_conclusion_layer(
            df_all=df_all,
            df_current=df,
            morphology_col=morphology_col,
            subset_name=subset_name,
        )
    except Exception as e:
        st.error(f"Could not render conclusion guide: {repr(e)}")


tabs = st.tabs(
    [
        "1. Story",
        "2. Morphology map",
        "3. What defines the clusters?",
        "4. QC validation",
        "5. REM / dissociation focus",
        "6. PD vs WT / week",
    ]
)

with tabs[0]:
    st.subheader("Analysis story after EMG burst detection")

    st.markdown(
        """
        **Pipeline**

        Detected EMG episodes  
        ↓  
        Extract morphology features: duration, amplitude, burstiness, shape, background tone, density  
        ↓  
        Unsupervised morphology map: PCA / UMAP / GMM  
        ↓  
        Rename clusters into interpretable motor phenotypes  
        ↓  
        Overlay biological information: REM relevance, REM–Wake dissociation, QC labels, PD vs WT, week 2 vs late week

        **What to say in the meeting**

        1. We are no longer only counting EMG bursts. We are asking what kind of motor events they are.
        2. The unsupervised model only sees EMG morphology. It does not know QC status, genotype, or REM labels.
        3. After clustering, we ask whether RBD-like or dissociation-positive events fall into specific morphology regions.
        4. Genotype/week conclusions should be based on mouse-level plots, not only event-level counts.
        """
    )

    st.warning(
        """
        Interpretation rule: UMAP is for visualization and cluster interpretation.
        PD vs WT claims should rely on mouse-level summaries and REM-normalized burden.
        """
    )

with tabs[1]:
    st.subheader("Morphology map")

    color_options = [
        c for c in [
            "broad_morphology_name",
            "cluster_short_name",
            "cluster_name",
            "qc_status",
            "primary_category",
            "group",
            "week",
        ]
        if c in df.columns
    ]

    col1, col2 = st.columns(2)

    with col1:
        color_col = st.selectbox("Color UMAP by", color_options, index=0)

    with col2:
        highlight_options = ["None", "REM-relevant", "Dissociation-positive"]
        if "qc_status" in df.columns:
            for status in sorted(df["qc_status"].dropna().unique()):
                highlight_options.append(f"QC: {status}")

        highlight_kind = st.selectbox("Highlight instead of coloring", highlight_options, index=0)

    fig = plot_umap(df, color_col=color_col, highlight_kind=highlight_kind)
    st.pyplot(fig)

    if st.button("Save UMAP plot"):
        save_current_figure(fig, "01_umap_morphology_map.png")

    plt.close(fig)

    st.caption(
        "Each point is one EMG event. Points close together have similar EMG morphology."
    )

with tabs[2]:
    st.subheader("What defines each cluster?")

    cards = cluster_summary_cards(df_all)

    st.markdown("### Descriptive cluster names")
    st.dataframe(cards, use_container_width=True)

    label_col = st.selectbox(
        "Explain features by",
        ["cluster_name", "cluster_short_name", "broad_morphology_name"],
        index=0,
    )

    z = feature_z_profiles(df_all, label_col)

    if z.empty:
        st.warning("No feature profiles available.")
    else:
        selected_label = st.selectbox("Selected cluster / morphology", list(z.index))
        fig = plot_feature_driver_bars(z, selected_label)

        st.pyplot(fig)

        if st.button("Save feature-driver plot"):
            save_current_figure(fig, "02_feature_driver_plot.png")

        plt.close(fig)

        st.markdown("### Plain-language feature dictionary")
        feature_help_df = pd.DataFrame(
            [{"feature": k, "meaning": v} for k, v in FEATURE_HELP.items()]
        )
        st.dataframe(feature_help_df, use_container_width=True)

    st.info(
        """
        Suggested wording: sustained/tonic clusters are characterized by longer duration,
        more total active EMG, and stronger amplitude. Phasic clusters are shorter and more compact.
        Clustered high-tone events show repeated bursts and elevated background tone.
        """
    )

with tabs[3]:
    st.subheader("QC validation")

    st.markdown(
        """
        This asks whether the unsupervised morphology classes are meaningful for manual/video QC.
        The important plot is QC outcome by morphology among reviewed events only.
        """
    )

    fig, qc_yield = plot_qc_by_morphology(df, morphology_col)
    st.pyplot(fig)

    if st.button("Save QC validation plot"):
        save_current_figure(fig, "03_qc_validation_by_morphology.png")

    plt.close(fig)

    if len(qc_yield):
        st.markdown("### QC yield by morphology")
        st.dataframe(qc_yield.round(1), use_container_width=True)

    st.warning(
        "Do not interpret the not-reviewed pool as negative. QC-based conclusions should use reviewed events only."
    )

with tabs[4]:
    st.subheader("REM / dissociation focus")

    st.markdown(
        """
        This is the most RBD-specific view. It asks whether REM-relevant, dissociation-positive,
        and QC possible RBD-like events are enriched in phasic, clustered, or sustained/tonic morphology.
        """
    )

    fig, subset_table = plot_morphology_by_subsets(df_all, morphology_col)
    st.pyplot(fig)

    if st.button("Save REM/dissociation morphology plot"):
        save_current_figure(fig, "04_rem_dissociation_morphology_distribution.png")

    plt.close(fig)

    with st.expander("Underlying counts for this plot"):
        st.dataframe(subset_table, use_container_width=True)

    if "candidate_rbd_like_dissociation_recomputed" in df_all.columns:
        diss = df_all[df_all["candidate_rbd_like_dissociation_recomputed"].fillna(False)]
        if len(diss) and morphology_col in diss.columns:
            top = diss[morphology_col].value_counts(normalize=True).mul(100).round(1)
            st.markdown("### Dissociation-positive morphology distribution")
            st.write(top)

    st.info(
        """
        Suggested wording: if dissociation-positive or QC possible RBD-like events are enriched
        in sustained/tonic or clustered high-tone morphology, this supports the idea that
        RBD-like activity is not a single burst type but contains distinct motor phenotypes.
        """
    )

with tabs[5]:
    st.subheader("PD vs WT / week comparison")

    st.markdown(
        """
        Use these plots carefully. The stacked bars show group/week morphology composition.
        The difference heatmap makes the contrast explicit. Mouse-level mode is safer for biological interpretation.
        """
    )

    unit = st.radio(
        "Composition unit",
        ["mouse-level", "event-level"],
        index=0,
        horizontal=True,
        help="Mouse-level is preferred for biological conclusions.",
    )

    comp = composition_matrix(df, morphology_col, unit=unit)
    order = ordered_group_week_labels(df)

    fig = plot_stacked_composition(
        comp,
        title=f"Morphology composition by group/week ({unit}, subset: {subset_name})",
        order=order,
    )
    st.pyplot(fig)

    if st.button("Save group/week composition plot"):
        save_current_figure(fig, "05_group_week_morphology_composition.png")

    plt.close(fig)

    diff = contrast_heatmap(comp)

    fig = plot_difference_heatmap(
        diff,
        title=f"Group/week differences in morphology composition ({unit})",
    )
    st.pyplot(fig)

    if st.button("Save difference heatmap"):
        save_current_figure(fig, "06_group_week_difference_heatmap.png")

    plt.close(fig)

    morph_values = sorted(df[morphology_col].dropna().unique())

    if morph_values:
        selected_morph = st.selectbox(
            "Mouse-level dot plot for one morphology",
            morph_values,
            index=0,
        )

        fig = plot_mouse_level_dot(df, morphology_col, selected_morph)
        st.pyplot(fig)

        if st.button("Save mouse-level dot plot"):
            save_current_figure(fig, "07_mouse_level_morphology_dotplot.png")

        plt.close(fig)

    st.warning(
        """
        Suggested conclusion language: group/week structure is visible, but genotype/progression
        claims should be made after mouse-level REM-normalized morphology burden is computed.
        """
    )

# -------------------------------------------------------------------------
# Export current filtered data
# -------------------------------------------------------------------------

with st.expander("Export / representative events"):
    st.download_button(
        "Download currently displayed event table",
        data=df.to_csv(index=False),
        file_name="supervisor_current_filtered_emg_morphology_events.csv",
        mime="text/csv",
    )

    sort_col = first_existing_col(
        df,
        [
            "rbd_dissociation_score_recomputed",
            "emg_rms_z_max",
            "morph_duration_sec",
            "active_fraction_in_event",
        ],
    )

    if sort_col is not None:
        st.markdown("### Top representative events in current view")
        show_cols = [
            "qc_event_id",
            "group",
            "week",
            "mouse_id",
            "segment_id",
            "qc_status",
            "primary_category",
            "cluster_name",
            "broad_morphology_name",
            "P_REM_EEGonly",
            "P_REM_FULL",
            "delta_REM",
            "rbd_dissociation_score_recomputed",
            "morph_duration_sec",
            "emg_rms_z_max",
            "active_fraction_in_event",
            "n_subbursts",
        ]
        show_cols = [c for c in show_cols if c in df.columns]

        st.dataframe(
            df.sort_values(sort_col, ascending=False)[show_cols].head(25),
            use_container_width=True,
        )
