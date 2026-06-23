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


FEATURE_EXPLANATIONS = {
    "morph_duration_sec": "event duration",
    "emg_rms_z_max": "maximum EMG amplitude",
    "emg_rms_z_mean": "average EMG amplitude",
    "active_fraction_in_event": "fraction of event that is active",
    "n_subbursts": "number of sub-bursts",
    "total_active_duration_sec": "total active EMG time",
    "rms_peak_to_mean_ratio": "peakiness / sharpness",
    "background_rms_median": "background EMG tone",
    "pre_event_rms_median": "pre-event tone",
    "post_event_rms_median": "post-event tone",
    "rms_peak_rate_per_sec": "peak rate",
    "rms_peak_regularity_score": "regularity of peaks",
    "emg_event_to_background_rms_ratio": "event-to-background contrast",
    "emg_spectral_high_fraction_50_150": "high-frequency EMG fraction",
}


MORPHOLOGY_EXPLANATIONS = {
    "phasic_or_uncertain": (
        "Very brief or ambiguous EMG events. These may include small twitches, "
        "low-amplitude bursts, or events that are difficult to interpret from EMG alone."
    ),
    "phasic": (
        "Brief twitch-like EMG events. These are short, compact activations, "
        "usually with one dominant burst."
    ),
    "clustered_phasic_high_tone": (
        "Repeated phasic bursts occurring on elevated background EMG tone. "
        "These may represent clustered twitching or unstable motor activation."
    ),
    "sustained_tonic": (
        "Longer and more complex EMG activation. These events have longer duration, "
        "more sub-bursts, and often higher amplitude; this is the class most consistent "
        "with sustained/tumbling-like motor activity."
    ),
}


REM_SUBSETS = {
    "All events": lambda df: pd.Series(True, index=df.index),
    "REM-relevant automatic": lambda df: df.get("REM_relevant_auto", False),
    "EEG-only REM-like": lambda df: df.get("EEGonly_PREM_ge_060", False),
    "Dissociation-positive": lambda df: df.get("candidate_rbd_like_dissociation_recomputed", False),
    "QC possible RBD-like": lambda df: df.get("qc_status", "").eq("possible_RBD_like"),
    "QC reviewed only": lambda df: df.get("qc_status", "").ne("not_reviewed"),
}


@st.cache_data(show_spinner=False)
def load_enhanced_umap_table(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "qc_status" not in df.columns:
        df["qc_status"] = "not_reviewed"

    if "emg_morphology_broad_label" not in df.columns and "emg_morphology_cluster" in df.columns:
        broad_map = {
            0: "phasic_or_uncertain",
            1: "phasic",
            2: "sustained_tonic",
            3: "phasic",
            4: "clustered_phasic_high_tone",
            5: "sustained_tonic",
        }
        cluster = pd.to_numeric(df["emg_morphology_cluster"], errors="coerce").astype("Int64")
        df["emg_morphology_broad_label"] = cluster.map(broad_map)

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


def crosstab_count_and_percent(df, row_col, col_col):
    count = pd.crosstab(df[row_col], df[col_col], dropna=False)
    pct = pd.crosstab(df[row_col], df[col_col], normalize="index", dropna=False) * 100
    return count, pct


def plot_umap(df, color_col, title):
    fig, ax = plt.subplots(figsize=(8.5, 6.3))

    if color_col not in df.columns:
        ax.scatter(df["umap1"], df["umap2"], s=8, alpha=0.65)
    else:
        vals = df[color_col].fillna("NA").astype(str)
        cats = sorted(vals.unique())

        for cat in cats:
            sub = df[vals == cat]
            ax.scatter(
                sub["umap1"],
                sub["umap2"],
                s=9,
                alpha=0.72,
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

    return fig


def feature_driver_table(df, label_col="emg_morphology_broad_label"):
    feature_cols = [c for c in FEATURE_EXPLANATIONS if c in df.columns]

    if not feature_cols or label_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    med = (
        df.groupby(label_col)[feature_cols]
        .median(numeric_only=True)
        .sort_index()
    )

    global_med = df[feature_cols].median(numeric_only=True)
    global_std = df[feature_cols].std(numeric_only=True).replace(0, np.nan)

    z = (med - global_med) / global_std

    rows = []

    for label in z.index:
        high = z.loc[label].sort_values(ascending=False).head(3)
        low = z.loc[label].sort_values(ascending=True).head(2)

        rows.append({
            "morphology_label": label,
            "plain_language_description": MORPHOLOGY_EXPLANATIONS.get(label, ""),
            "main_high_features": ", ".join(
                [FEATURE_EXPLANATIONS.get(c, c) for c in high.index]
            ),
            "main_low_features": ", ".join(
                [FEATURE_EXPLANATIONS.get(c, c) for c in low.index]
            ),
        })

    explain = pd.DataFrame(rows)

    readable_med = med.rename(columns=FEATURE_EXPLANATIONS)

    return explain, readable_med


def reviewed_qc_yield_table(df):
    if "qc_status" not in df.columns or "emg_morphology_broad_label" not in df.columns:
        return pd.DataFrame()

    reviewed = df[df["qc_status"].ne("not_reviewed")].copy()

    if len(reviewed) == 0:
        return pd.DataFrame()

    tab = pd.crosstab(
        reviewed["emg_morphology_broad_label"],
        reviewed["qc_status"],
        dropna=False,
    )

    tab["reviewed_total"] = tab.sum(axis=1)

    if "possible_RBD_like" in tab.columns:
        tab["possible_RBD_yield_pct"] = (
            100 * tab["possible_RBD_like"] / tab["reviewed_total"].replace(0, np.nan)
        )
    else:
        tab["possible_RBD_yield_pct"] = 0.0

    return tab.sort_values("possible_RBD_yield_pct", ascending=False)


def morphology_distribution_row(df, mask, label):
    sub = df[mask].copy()

    if len(sub) == 0:
        return {"subset": label, "n": 0}

    counts = sub["emg_morphology_broad_label"].value_counts()
    total = counts.sum()

    row = {"subset": label, "n": int(total)}

    for morph in sorted(df["emg_morphology_broad_label"].dropna().unique()):
        row[f"{morph}_n"] = int(counts.get(morph, 0))
        row[f"{morph}_pct"] = 100 * counts.get(morph, 0) / total if total else np.nan

    return row


def render_unsupervised_morphology_dashboard(filtered_events=None):
    with st.expander("Unsupervised EMG morphology interpretation dashboard", expanded=False):
        st.markdown("## Unsupervised EMG morphology interpretation")

        st.info(
            """
            **How to read this analysis:** each dot is one EMG event. Events that are close
            together have similar morphology: similar duration, amplitude, burstiness,
            background tone, and shape. The unsupervised model was not told whether an
            event was RBD-like, artifact, transition, PD, or WT. Those labels are only
            added afterwards to interpret the morphology map.
            """
        )

        umap_path = st.text_input(
            "Enhanced UMAP/QC table",
            str(DEFAULT_ENHANCED_UMAP),
            key="unsup_interpretation_umap_path",
        )

        df = load_enhanced_umap_table(umap_path)

        if len(df) == 0:
            st.warning("Could not load the enhanced UMAP/QC table.")
            return

        use_current_filter = st.checkbox(
            "Use current sidebar/event filters for this dashboard",
            value=True,
            key="unsup_interpretation_use_current_filter",
        )

        if use_current_filter:
            df = restrict_to_current_filter(df, filtered_events)

        subset_name = st.selectbox(
            "Biological subset",
            list(REM_SUBSETS.keys()),
            index=0,
            key="unsup_interpretation_subset",
        )

        mask = REM_SUBSETS[subset_name](df)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=df.index)

        df_sub = df[mask.fillna(False)].copy()

        if len(df_sub) == 0:
            st.warning("No events in this selected subset.")
            return

        n_total = len(df_sub)
        n_reviewed = int(df_sub["qc_status"].ne("not_reviewed").sum()) if "qc_status" in df_sub.columns else 0
        n_possible = int(df_sub["qc_status"].eq("possible_RBD_like").sum()) if "qc_status" in df_sub.columns else 0
        n_rem_rel = int(df_sub.get("REM_relevant_auto", pd.Series(False, index=df_sub.index)).sum())
        n_diss = int(df_sub.get("candidate_rbd_like_dissociation_recomputed", pd.Series(False, index=df_sub.index)).sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Events shown", n_total)
        c2.metric("QC reviewed", n_reviewed)
        c3.metric("QC possible RBD-like", n_possible)
        c4.metric("REM-relevant", n_rem_rel)
        c5.metric("Dissociation-positive", n_diss)

        tabs = st.tabs(
            [
                "Map",
                "Feature drivers",
                "QC validation",
                "REM/RBD focus",
                "Conclusion checklist",
            ]
        )

        with tabs[0]:
            st.markdown("### Morphology map")

            color_options = [
                c for c in [
                    "emg_morphology_broad_label",
                    "emg_morphology_fine_label",
                    "emg_morphology_cluster",
                    "qc_status",
                    "primary_category",
                    "group",
                    "week",
                ]
                if c in df_sub.columns
            ]

            color_col = st.selectbox(
                "Color dots by",
                color_options,
                index=0,
                key="unsup_interpretation_color_col",
            )

            fig = plot_umap(
                df_sub,
                color_col,
                f"UMAP of EMG morphology colored by {color_col} ({subset_name})",
            )
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "UMAP is a visualization of local similarity. The axes themselves do not "
                "have direct biological units; use the feature-driver table to interpret "
                "what separates regions."
            )

        with tabs[1]:
            st.markdown("### What features motivate the morphology labels?")

            explain, med = feature_driver_table(df_sub)

            if len(explain):
                st.write("Plain-language explanation")
                st.dataframe(explain, use_container_width=True)

            if len(med):
                st.write("Median feature values by morphology class")
                st.dataframe(med.round(3), use_container_width=True)

            st.markdown("### Recommended wording")
            st.markdown(
                """
                The model groups events based on their EMG shape.  
                For example, sustained/tonic events are mainly driven by longer duration,
                more sub-bursts, longer active EMG time, and higher amplitude.  
                Phasic events are shorter and more compact.  
                Clustered/high-tone events are driven by repeated bursts and elevated
                surrounding EMG tone.
                """
            )

        with tabs[2]:
            st.markdown("### Do QC labels fall into different morphology classes?")

            if "qc_status" not in df_sub.columns:
                st.warning("No QC status column available.")
            else:
                count, pct = crosstab_count_and_percent(
                    df_sub,
                    "qc_status",
                    "emg_morphology_broad_label",
                )

                st.write("QC status × morphology counts")
                st.dataframe(count, use_container_width=True)

                st.write("QC status × morphology row %")
                st.dataframe(pct.round(1), use_container_width=True)

                yield_tab = reviewed_qc_yield_table(df_sub)
                if len(yield_tab):
                    st.write("Among reviewed events: possible RBD-like yield by morphology")
                    st.dataframe(yield_tab.round(1), use_container_width=True)

                st.caption(
                    "For conclusions about QC labels, focus on reviewed events. "
                    "The not_reviewed pool should not be interpreted as negative evidence."
                )

        with tabs[3]:
            st.markdown("### REM/RBD-focused morphology distributions")

            rows = []

            rows.append(
                morphology_distribution_row(
                    df_sub,
                    pd.Series(True, index=df_sub.index),
                    f"Current subset: {subset_name}",
                )
            )

            if "REM_relevant_auto" in df_sub.columns:
                rows.append(
                    morphology_distribution_row(
                        df_sub,
                        df_sub["REM_relevant_auto"].fillna(False),
                        "REM-relevant automatic",
                    )
                )

            if "candidate_rbd_like_dissociation_recomputed" in df_sub.columns:
                rows.append(
                    morphology_distribution_row(
                        df_sub,
                        df_sub["candidate_rbd_like_dissociation_recomputed"].fillna(False),
                        "Dissociation-positive candidates",
                    )
                )

            if "qc_status" in df_sub.columns:
                rows.append(
                    morphology_distribution_row(
                        df_sub,
                        df_sub["qc_status"].eq("possible_RBD_like"),
                        "QC possible RBD-like",
                    )
                )

            dist = pd.DataFrame(rows)
            st.dataframe(dist.round(1), use_container_width=True)

            st.markdown("### What this answers")
            st.markdown(
                """
                This table asks whether REM-like or QC-confirmed RBD-like events are mostly
                brief/phasic, clustered, or sustained/tonic. This directly addresses the
                question of whether there are different types of RBD-like motor activity.
                """
            )

        with tabs[4]:
            st.markdown("### Conclusion checklist")

            # Use the full selected data for quantitative statements.
            possible = df_sub[df_sub.get("qc_status", "").eq("possible_RBD_like")]
            diss = df_sub[df_sub.get("candidate_rbd_like_dissociation_recomputed", False).fillna(False)]

            rows = []

            rows.append({
                "question": "Did unsupervised learning find interpretable EMG phenotypes?",
                "current_answer": (
                    "Yes. The morphology labels correspond to intuitive feature profiles: "
                    "brief/phasic, clustered high-tone, and sustained/tonic events."
                ),
                "confidence": "High for morphology description",
                "what_to_check_next": "Inspect representative raw traces/video from each morphology label.",
            })

            if len(possible) > 0:
                pct_sust = 100 * possible["emg_morphology_broad_label"].eq("sustained_tonic").mean()
                rows.append({
                    "question": "Are QC possible RBD-like events enriched in a morphology class?",
                    "current_answer": f"Yes, in this subset {pct_sust:.1f}% are sustained/tonic.",
                    "confidence": "Moderate, limited by QC coverage",
                    "what_to_check_next": "Increase QC coverage and report reviewed-event yields by morphology.",
                })

            if len(diss) > 0:
                pct_sust = 100 * diss["emg_morphology_broad_label"].eq("sustained_tonic").mean()
                rows.append({
                    "question": "Are dissociation-positive events enriched in sustained/tonic morphology?",
                    "current_answer": f"Yes, in this subset {pct_sust:.1f}% are sustained/tonic.",
                    "confidence": "Moderate exploratory",
                    "what_to_check_next": "Normalize by EEG-only REM time at mouse level.",
                })

            rows.append({
                "question": "Can we conclude PD vs WT differences from this plot alone?",
                "current_answer": (
                    "Not yet. The UMAP is descriptive. Genotype/week claims should use "
                    "mouse-level rates normalized by REM time and QC coverage."
                ),
                "confidence": "Low from UMAP alone",
                "what_to_check_next": "Compute morphology-specific REM burden per mouse/week.",
            })

            conclusion = pd.DataFrame(rows)
            st.dataframe(conclusion, use_container_width=True)

            st.warning(
                """
                Suggested interpretation: use UMAP/PCA to show the morphology atlas,
                use feature-driver tables to explain the clusters, and use mouse-level
                normalized metrics to make biological PD/RBD conclusions.
                """
            )

        st.download_button(
            "Download displayed unsupervised morphology table",
            data=df_sub.to_csv(index=False),
            file_name="displayed_unsupervised_morphology_subset.csv",
            mime="text/csv",
        )
