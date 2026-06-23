from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


BASE = Path(
    "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/"
    "KjærbyLab/Project_PD_RBD_Katia/Data/prepared_data/manifests"
)

DEFAULT_ENHANCED_UMAP = (
    BASE
    / "EMG_unsupervised_morphology"
    / "refined_morphology_clustering"
    / "umap"
    / "qc_dissociation_enhanced"
    / "emg_morphology_umap_embedding_with_qc_and_dissociation.csv"
)


# =============================================================================
# INTERPRETABLE CLUSTER NAMES
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
    "emg_abs_z_mean": "Mean rectified EMG",
    "emg_abs_z_max": "Max rectified EMG",
    "emg_abs_z_p95": "95th pct rectified EMG",
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
    "rms_peak_interval_mean_sec": "Mean peak interval",
    "rms_peak_interval_std_sec": "Peak interval variability",
    "rms_peak_interval_cv": "Peak irregularity",
    "rms_peak_regularity_score": "Peak regularity",
    "rms_peak_time_fraction": "Peak timing in event",
    "rms_rise_time_sec": "Rise time",
    "rms_decay_time_sec": "Decay time",
    "rms_rise_decay_ratio": "Rise/decay ratio",
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
    "Pre-event tone": "Muscle tone immediately before the event.",
    "Post-event tone": "Muscle tone immediately after the event.",
    "Number of sub-bursts": "How many smaller bursts are contained inside the event.",
    "Active fraction": "How much of the event is occupied by active EMG.",
    "Peakiness": "Whether the event is sharp/peaky or broad/sustained.",
    "Peak irregularity": "How irregular the intervals between peaks are.",
    "Peak regularity": "How regular the repeated peaks are.",
    "High-frequency EMG fraction": "How much of the EMG power is in a higher-frequency range.",
}


# =============================================================================
# LOAD / PREP
# =============================================================================

@st.cache_data(show_spinner=False)
def load_umap_table(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "qc_status" not in df.columns:
        df["qc_status"] = "not_reviewed"
    else:
        df["qc_status"] = df["qc_status"].fillna("not_reviewed")

    if "emg_morphology_cluster" in df.columns:
        cluster = pd.to_numeric(df["emg_morphology_cluster"], errors="coerce").astype("Int64")
        df["cluster_name"] = cluster.map(CLUSTER_NAME_MAP)
        df["cluster_short_name"] = cluster.map(CLUSTER_SHORT_MAP)
        df["broad_morphology_name"] = cluster.map(CLUSTER_BROAD_MAP)
        df["cluster_explanation"] = cluster.map(CLUSTER_EXPLANATION_MAP)

    if "broad_morphology_name" not in df.columns and "emg_morphology_broad_label" in df.columns:
        df["broad_morphology_name"] = df["emg_morphology_broad_label"].astype(str)

    if "candidate_rbd_like_dissociation_recomputed" not in df.columns:
        df["candidate_rbd_like_dissociation_recomputed"] = False

    if "REM_relevant_auto" not in df.columns:
        rem_categories = [
            "stable_REM_EMG_burst",
            "EMG_suppressed_REM",
            "mixed_REM_Wake_transition",
        ]
        if "primary_category" in df.columns:
            df["REM_relevant_auto"] = df["primary_category"].isin(rem_categories)
        else:
            df["REM_relevant_auto"] = False

    return df


def restrict_to_current_filter(umap_df, filtered_events):
    if filtered_events is None or len(filtered_events) == 0:
        return umap_df

    if "stable_event_key" in umap_df.columns and "stable_event_key" in filtered_events.columns:
        keys = set(filtered_events["stable_event_key"].astype(str))
        return umap_df[umap_df["stable_event_key"].astype(str).isin(keys)].copy()

    if "qc_event_id" in umap_df.columns and "qc_event_id" in filtered_events.columns:
        ids = set(pd.to_numeric(filtered_events["qc_event_id"], errors="coerce").dropna().astype(int))
        event_ids = pd.to_numeric(umap_df["qc_event_id"], errors="coerce")
        return umap_df[event_ids.isin(ids)].copy()

    return umap_df


def get_group_week_label(df):
    out = df.copy()
    out["week_int"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    out["group_week"] = out["group"].astype(str) + " W" + out["week_int"].astype(str)
    return out


def ordered_group_week_labels(df):
    order = []
    for g in ["WT", "PD"]:
        if "group" not in df.columns or "week" not in df.columns:
            continue
        weeks = sorted(pd.to_numeric(df.loc[df["group"] == g, "week"], errors="coerce").dropna().unique())
        for w in weeks:
            order.append(f"{g} W{int(w)}")

    existing = set(df.get("group_week", pd.Series(dtype=str)).dropna().unique())
    order = [x for x in order if x in existing]

    extras = sorted(existing - set(order))
    return order + extras


# =============================================================================
# PLOTS
# =============================================================================

def plot_umap(df, color_col, highlight_col=None):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    if "umap1" not in df.columns or "umap2" not in df.columns:
        ax.text(0.5, 0.5, "No UMAP columns found", ha="center", va="center")
        return fig

    if highlight_col and highlight_col in df.columns:
        mask = df[highlight_col].fillna(False).astype(bool)

        ax.scatter(
            df.loc[~mask, "umap1"],
            df.loc[~mask, "umap2"],
            s=6,
            alpha=0.15,
            label="Other",
        )
        ax.scatter(
            df.loc[mask, "umap1"],
            df.loc[mask, "umap2"],
            s=18,
            alpha=0.85,
            label=highlight_col,
        )
        ax.legend(loc="best", fontsize=8)
    else:
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
    df = get_group_week_label(df)

    if category_col not in df.columns:
        return pd.DataFrame()

    if unit == "event-level":
        tab = pd.crosstab(df["group_week"], df[category_col], normalize="index") * 100
        return tab

    if "mouse_id" not in df.columns:
        tab = pd.crosstab(df["group_week"], df[category_col], normalize="index") * 100
        return tab

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

    if order:
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

    # Parse available group/week labels.
    labels = list(comp.index)

    def find_label(group, week_kind):
        candidates = [x for x in labels if x.startswith(group + " W")]
        if not candidates:
            return None

        weeks = []
        for x in candidates:
            try:
                weeks.append((int(x.split("W")[-1]), x))
            except Exception:
                pass

        if not weeks:
            return None

        weeks = sorted(weeks)

        if week_kind == "early":
            return weeks[0][1]
        if week_kind == "late":
            return weeks[-1][1]
        return None

    wt_early = find_label("WT", "early")
    wt_late = find_label("WT", "late")
    pd_early = find_label("PD", "early")
    pd_late = find_label("PD", "late")

    rows = {}

    def add_contrast(name, a, b):
        if a in comp.index and b in comp.index:
            rows[name] = comp.loc[a] - comp.loc[b]

    add_contrast("PD late - WT late", pd_late, wt_late)
    add_contrast("PD early - WT early", pd_early, wt_early)
    add_contrast("PD late - PD early", pd_late, pd_early)
    add_contrast("WT late - WT early", wt_late, wt_early)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).T
    return out


def plot_difference_heatmap(diff, title):
    fig, ax = plt.subplots(figsize=(9.8, 4.8))

    if diff.empty:
        ax.text(0.5, 0.5, "No valid contrasts", ha="center", va="center")
        return fig

    max_abs = np.nanmax(np.abs(diff.values))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1

    im = ax.imshow(diff.values, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

    ax.set_yticks(np.arange(diff.shape[0]))
    ax.set_yticklabels(diff.index)

    ax.set_xticks(np.arange(diff.shape[1]))
    ax.set_xticklabels(diff.columns, rotation=45, ha="right")

    ax.set_title(title)

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            val = diff.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Difference in percentage points")

    fig.tight_layout()
    return fig


def plot_qc_by_morphology(df, morphology_col):
    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    if "qc_status" not in df.columns or morphology_col not in df.columns:
        ax.text(0.5, 0.5, "Missing QC or morphology column", ha="center", va="center")
        return fig

    reviewed = df[df["qc_status"].ne("not_reviewed")].copy()

    if len(reviewed) == 0:
        ax.text(0.5, 0.5, "No reviewed QC events in current subset", ha="center", va="center")
        return fig

    tab = pd.crosstab(reviewed[morphology_col], reviewed["qc_status"], normalize="index") * 100
    tab.plot(kind="bar", stacked=True, ax=ax, width=0.75)

    ax.set_ylabel("% within morphology class")
    ax.set_title("QC outcome by morphology class, reviewed events only")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


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


def cluster_cards(df):
    rows = []

    if "emg_morphology_cluster" not in df.columns:
        return pd.DataFrame()

    z = feature_z_profiles(df, "cluster_name")

    for clust in sorted(pd.to_numeric(df["emg_morphology_cluster"], errors="coerce").dropna().unique()):
        clust = int(clust)
        name = CLUSTER_NAME_MAP.get(clust, f"Cluster {clust}")
        short = CLUSTER_SHORT_MAP.get(clust, f"cluster_{clust}")
        broad = CLUSTER_BROAD_MAP.get(clust, "")
        explanation = CLUSTER_EXPLANATION_MAP.get(clust, "")

        if name in z.index:
            s = z.loc[name].dropna()
            high = s.sort_values(ascending=False).head(3)
            low = s.sort_values(ascending=True).head(2)

            high_text = ", ".join(high.index)
            low_text = ", ".join(low.index)
        else:
            high_text = ""
            low_text = ""

        n = int((pd.to_numeric(df["emg_morphology_cluster"], errors="coerce") == clust).sum())

        rows.append({
            "cluster": clust,
            "new_name": name,
            "short_name": short,
            "broad_class": broad,
            "n_events_in_current_view": n,
            "features_high": high_text,
            "features_low": low_text,
            "plain_explanation": explanation,
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_unsupervised_visual_dashboard(filtered_events=None):
    with st.expander("Visual unsupervised EMG morphology dashboard", expanded=True):
        st.markdown("## Visual unsupervised EMG morphology dashboard")

        st.info(
            """
            **Plain explanation:** each point is one EMG event. Events close together have
            similar morphology, based on duration, amplitude, burstiness, background tone,
            and shape. The unsupervised model was not given QC labels or genotype labels;
            those are overlaid afterwards to interpret the map.
            """
        )

        path = st.text_input(
            "Enhanced UMAP/QC table",
            str(DEFAULT_ENHANCED_UMAP),
            key="visual_unsup_umap_path",
        )

        df = load_umap_table(path)

        if len(df) == 0:
            st.warning("Could not load enhanced UMAP/QC table.")
            return

        use_current_filter = st.checkbox(
            "Use current sidebar filters",
            value=True,
            key="visual_unsup_use_current_filter",
        )

        if use_current_filter:
            df = restrict_to_current_filter(df, filtered_events)

        if len(df) == 0:
            st.warning("No events after current filters.")
            return

        subset_options = {
            "All events": pd.Series(True, index=df.index),
            "REM-relevant automatic": df.get("REM_relevant_auto", pd.Series(False, index=df.index)),
            "Dissociation-positive": df.get("candidate_rbd_like_dissociation_recomputed", pd.Series(False, index=df.index)),
            "QC reviewed only": df.get("qc_status", pd.Series("not_reviewed", index=df.index)).ne("not_reviewed"),
            "QC possible RBD-like": df.get("qc_status", pd.Series("", index=df.index)).eq("possible_RBD_like"),
            "Stable REM EMG burst": df.get("primary_category", pd.Series("", index=df.index)).eq("stable_REM_EMG_burst"),
            "Mixed REM-Wake transition": df.get("primary_category", pd.Series("", index=df.index)).eq("mixed_REM_Wake_transition"),
        }

        subset_name = st.selectbox(
            "Biological subset to display",
            list(subset_options.keys()),
            index=0,
            key="visual_unsup_subset",
        )

        mask = subset_options[subset_name].fillna(False).astype(bool)
        df_sub = df[mask].copy()

        if len(df_sub) == 0:
            st.warning("No events in this biological subset.")
            return

        metric_cols = st.columns(5)
        metric_cols[0].metric("Events", len(df_sub))
        metric_cols[1].metric("Mice", df_sub["mouse_id"].nunique() if "mouse_id" in df_sub.columns else "NA")
        metric_cols[2].metric("QC reviewed", int(df_sub.get("qc_status", pd.Series("not_reviewed", index=df_sub.index)).ne("not_reviewed").sum()))
        metric_cols[3].metric("QC RBD-like", int(df_sub.get("qc_status", pd.Series("", index=df_sub.index)).eq("possible_RBD_like").sum()))
        metric_cols[4].metric("Dissociation+", int(df_sub.get("candidate_rbd_like_dissociation_recomputed", pd.Series(False, index=df_sub.index)).sum()))

        tabs = st.tabs(
            [
                "1. Morphology map",
                "2. Group/week composition",
                "3. Difference heatmap",
                "4. QC validation",
                "5. Feature interpretation",
            ]
        )

        with tabs[0]:
            st.markdown("### 1. Morphology map")

            color_options = [
                c for c in [
                    "cluster_name",
                    "cluster_short_name",
                    "broad_morphology_name",
                    "qc_status",
                    "primary_category",
                    "group",
                    "week",
                ]
                if c in df_sub.columns
            ]

            col1, col2 = st.columns(2)

            with col1:
                color_col = st.selectbox(
                    "Color by",
                    color_options,
                    index=0,
                    key="visual_unsup_color_col",
                )

            with col2:
                highlight_options = [
                    "None",
                    "REM_relevant_auto",
                    "candidate_rbd_like_dissociation_recomputed",
                ]
                if "qc_status" in df_sub.columns:
                    for status in sorted(df_sub["qc_status"].dropna().unique()):
                        highlight_options.append(f"QC: {status}")

                highlight_choice = st.selectbox(
                    "Highlight",
                    highlight_options,
                    index=0,
                    key="visual_unsup_highlight",
                )

            highlight_col = None
            plot_df = df_sub.copy()

            if highlight_choice.startswith("QC: "):
                status = highlight_choice.replace("QC: ", "")
                highlight_col = f"highlight_qc_{status}"
                plot_df[highlight_col] = plot_df["qc_status"].eq(status)
            elif highlight_choice != "None":
                highlight_col = highlight_choice

            fig = plot_umap(plot_df, color_col=color_col, highlight_col=highlight_col)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Use this as a map of EMG event morphology. The axes are not directly biological; "
                "interpret regions using the feature interpretation tab."
            )

        with tabs[1]:
            st.markdown("### 2. Morphology composition by group/week")

            morphology_col = st.selectbox(
                "Morphology level",
                ["broad_morphology_name", "cluster_short_name", "cluster_name"],
                index=0,
                key="visual_unsup_morphology_col_comp",
            )

            unit = st.radio(
                "Composition unit",
                ["mouse-level", "event-level"],
                index=0,
                horizontal=True,
                key="visual_unsup_composition_unit",
                help=(
                    "Mouse-level is better for biological conclusions because one mouse "
                    "cannot dominate just by having many events."
                ),
            )

            comp = composition_matrix(df_sub, morphology_col, unit=unit)
            order = ordered_group_week_labels(get_group_week_label(df_sub))

            fig = plot_stacked_composition(
                comp,
                title=f"{morphology_col} composition by group/week ({unit}, {subset_name})",
                order=order,
            )
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "This plot shows whether PD/WT and week 2/week 21 differ in the types of EMG events."
            )

            morph_values = sorted(df_sub[morphology_col].dropna().unique())
            if morph_values:
                selected_morph = st.selectbox(
                    "Mouse-level dot plot for one morphology",
                    morph_values,
                    key="visual_unsup_selected_morph_dot",
                )

                fig = plot_mouse_level_dot(df_sub, morphology_col, selected_morph)
                st.pyplot(fig)
                plt.close(fig)

        with tabs[2]:
            st.markdown("### 3. Difference heatmap")

            morphology_col = st.selectbox(
                "Morphology level for contrasts",
                ["broad_morphology_name", "cluster_short_name", "cluster_name"],
                index=0,
                key="visual_unsup_morphology_col_diff",
            )

            unit = st.radio(
                "Contrast unit",
                ["mouse-level", "event-level"],
                index=0,
                horizontal=True,
                key="visual_unsup_diff_unit",
            )

            comp = composition_matrix(df_sub, morphology_col, unit=unit)
            diff = contrast_heatmap(comp)

            fig = plot_difference_heatmap(
                diff,
                title=f"Group/week differences in morphology composition ({unit})",
            )
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                """
                Positive values mean the morphology is more common in the first group of the contrast.
                Example: `PD late - WT late = +10` means PD late has 10 percentage points more of that
                morphology than WT late.
                """
            )

        with tabs[3]:
            st.markdown("### 4. QC validation: where do reviewed labels fall?")

            morphology_col = st.selectbox(
                "Morphology level for QC plot",
                ["broad_morphology_name", "cluster_short_name", "cluster_name"],
                index=0,
                key="visual_unsup_morphology_col_qc",
            )

            fig = plot_qc_by_morphology(df_sub, morphology_col)
            st.pyplot(fig)
            plt.close(fig)

            if "qc_status" in df_sub.columns:
                reviewed = df_sub[df_sub["qc_status"].ne("not_reviewed")].copy()

                if len(reviewed):
                    yield_tab = pd.crosstab(
                        reviewed[morphology_col],
                        reviewed["qc_status"],
                        dropna=False,
                    )

                    yield_tab["reviewed_total"] = yield_tab.sum(axis=1)

                    if "possible_RBD_like" in yield_tab.columns:
                        yield_tab["possible_RBD_like_yield_%"] = (
                            100 * yield_tab["possible_RBD_like"] / yield_tab["reviewed_total"].replace(0, np.nan)
                        )

                    st.dataframe(yield_tab.round(1), use_container_width=True)

            st.caption(
                "This is useful for checking whether artifacts, transitions, and possible RBD-like events occupy different morphology classes."
            )

        with tabs[4]:
            st.markdown("### 5. What features define each cluster?")

            cards = cluster_cards(df_sub)

            if len(cards):
                st.markdown("#### Cluster names and explanations")
                st.dataframe(cards, use_container_width=True)

            label_col = st.selectbox(
                "Explain features for",
                ["cluster_name", "cluster_short_name", "broad_morphology_name"],
                index=0,
                key="visual_unsup_feature_label_col",
            )

            z = feature_z_profiles(df_sub, label_col)

            if len(z):
                selected_label = st.selectbox(
                    "Selected label",
                    list(z.index),
                    key="visual_unsup_selected_feature_label",
                )

                fig = plot_feature_driver_bars(z, selected_label)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("#### Plain feature meanings")
                feature_help_df = pd.DataFrame(
                    [
                        {"feature": k, "meaning": v}
                        for k, v in FEATURE_HELP.items()
                    ]
                )
                st.dataframe(feature_help_df, use_container_width=True)

            st.info(
                """
                Suggested wording: “The clusters are motivated by differences in event duration,
                amplitude, burst structure, active fraction, and background tone. Sustained/tonic
                clusters are characterized by longer duration and more total active EMG, while
                phasic clusters are shorter and more compact.”
                """
            )
