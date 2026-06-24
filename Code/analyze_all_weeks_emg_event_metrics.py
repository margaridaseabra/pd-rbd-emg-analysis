#!/usr/bin/env python3
"""
All-weeks EMG/RBD-like event metric analysis.

This script creates:
- mouse/week event metric table
- group/week summary table
- exploratory significance tests
- longitudinal plots

This is event-composition based. REM-normalized metrics should be added next.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


REM_RELEVANT_CATEGORIES = {
    "stable_REM_EMG_burst",
    "stable_EEG_REM_EMG_burst",
    "EMG_suppressed_REM",
    "candidate_EMG_suppressed_REM",
    "mixed_REM_Wake_transition",
    "REM_transition_EMG_burst",
}

STABLE_REM_CATEGORIES = {
    "stable_REM_EMG_burst",
    "stable_EEG_REM_EMG_burst",
}

SUPPRESSED_REM_CATEGORIES = {
    "EMG_suppressed_REM",
    "candidate_EMG_suppressed_REM",
}

TRANSITION_CATEGORIES = {
    "mixed_REM_Wake_transition",
    "REM_transition_EMG_burst",
}

WAKE_LIKE_CATEGORIES = {
    "wake_like_movement",
}


MOUSE_INFO = {
    1:  {"sex": "M", "genotype": "Null", "group": "WT"},
    2:  {"sex": "F", "genotype": "Null", "group": "WT"},
    3:  {"sex": "M", "genotype": "A53T", "group": "PD"},
    4:  {"sex": "F", "genotype": "A53T", "group": "PD"},
    5:  {"sex": "M", "genotype": "A53T", "group": "PD"},
    6:  {"sex": "F", "genotype": "A53T", "group": "PD"},
    7:  {"sex": "M", "genotype": "Null", "group": "WT"},
    8:  {"sex": "F", "genotype": "Null", "group": "WT"},
    9:  {"sex": "M", "genotype": "A53T", "group": "PD"},
    10: {"sex": "F", "genotype": "Null", "group": "WT"},
    11: {"sex": "M", "genotype": "Null", "group": "WT"},
    12: {"sex": "F", "genotype": "A53T", "group": "PD"},
    13: {"sex": "M", "genotype": "A53T", "group": "PD"},
    14: {"sex": "F", "genotype": "Null", "group": "WT"},
    15: {"sex": "M", "genotype": "A53T", "group": "PD"},
}


def ensure_numeric(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    p = pvals[valid]
    if len(p) == 0:
        return out

    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    temp = np.empty_like(adj)
    temp[order] = adj
    out[valid] = temp
    return out


def add_metadata_and_flags(df):
    df = df.copy()

    if "primary_category" not in df.columns:
        if "event_class" in df.columns:
            df["primary_category"] = df["event_class"]
        else:
            df["primary_category"] = "other_uncertain"

    df["mouse_id"] = pd.to_numeric(df["mouse_id"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Enforce correct genotype mapping
    df["sex"] = df["mouse_id"].map(
        lambda x: MOUSE_INFO[int(x)]["sex"] if pd.notna(x) and int(x) in MOUSE_INFO else ""
    )
    df["genotype"] = df["mouse_id"].map(
        lambda x: MOUSE_INFO[int(x)]["genotype"] if pd.notna(x) and int(x) in MOUSE_INFO else ""
    )
    df["group"] = df["mouse_id"].map(
        lambda x: MOUSE_INFO[int(x)]["group"] if pd.notna(x) and int(x) in MOUSE_INFO else ""
    )

    df["delta_REM"] = ensure_numeric(df, "delta_REM")
    df["P_REM_EEGonly"] = ensure_numeric(df, "P_REM_EEGonly")
    df["P_WAKE_EEGonly"] = ensure_numeric(df, "P_Wake_EEGonly")
    df["P_REM_FULL"] = ensure_numeric(df, "P_REM_FULL")

    if df["delta_REM"].isna().all() and {"P_REM_EEGonly", "P_REM_FULL"}.issubset(df.columns):
        df["delta_REM"] = df["P_REM_EEGonly"] - df["P_REM_FULL"]

    if "duration_sec_for_category" in df.columns:
        df["event_duration_sec"] = ensure_numeric(df, "duration_sec_for_category")
    elif "duration_sec" in df.columns:
        df["event_duration_sec"] = ensure_numeric(df, "duration_sec")
    elif {"start_sec", "end_sec"}.issubset(df.columns):
        df["event_duration_sec"] = ensure_numeric(df, "end_sec") - ensure_numeric(df, "start_sec")
    else:
        df["event_duration_sec"] = np.nan

    if "max_EMG_z" in df.columns:
        df["max_EMG_z_numeric"] = ensure_numeric(df, "max_EMG_z")
    elif "max_EMG_baseline_z" in df.columns:
        df["max_EMG_z_numeric"] = ensure_numeric(df, "max_EMG_baseline_z")
    else:
        df["max_EMG_z_numeric"] = np.nan

    df["is_REM_relevant"] = df["primary_category"].isin(REM_RELEVANT_CATEGORIES)
    df["is_stable_REM_EMG"] = df["primary_category"].isin(STABLE_REM_CATEGORIES)
    df["is_EMG_suppressed_REM"] = df["primary_category"].isin(SUPPRESSED_REM_CATEGORIES)
    df["is_transition_REM_Wake"] = df["primary_category"].isin(TRANSITION_CATEGORIES)
    df["is_wake_like"] = df["primary_category"].isin(WAKE_LIKE_CATEGORIES)

    df["is_dissociation_positive"] = (
        (df["delta_REM"] >= 0.25)
        & (df["P_REM_EEGonly"] >= 0.70)
        & (df["P_WAKE_EEGonly"] <= 0.30)
    )

    df["is_candidate_RBD_auto"] = (
        df["is_stable_REM_EMG"]
        | df["is_EMG_suppressed_REM"]
        | df["is_dissociation_positive"]
    )

    return df


def compute_mouse_week_metrics(df):
    group_cols = ["group", "genotype", "sex", "mouse_id", "week"]

    metrics = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_events=("primary_category", "size"),
            n_REM_relevant=("is_REM_relevant", "sum"),
            n_candidate_RBD_auto=("is_candidate_RBD_auto", "sum"),
            n_stable_REM_EMG=("is_stable_REM_EMG", "sum"),
            n_EMG_suppressed_REM=("is_EMG_suppressed_REM", "sum"),
            n_dissociation_positive=("is_dissociation_positive", "sum"),
            n_transition_REM_Wake=("is_transition_REM_Wake", "sum"),
            n_wake_like=("is_wake_like", "sum"),
            mean_delta_REM_all=("delta_REM", "mean"),
            median_delta_REM_all=("delta_REM", "median"),
            mean_P_REM_EEGonly_all=("P_REM_EEGonly", "mean"),
            mean_P_REM_FULL_all=("P_REM_FULL", "mean"),
            mean_event_duration_sec=("event_duration_sec", "mean"),
            median_event_duration_sec=("event_duration_sec", "median"),
            mean_max_EMG_z=("max_EMG_z_numeric", "mean"),
            median_max_EMG_z=("max_EMG_z_numeric", "median"),
        )
        .reset_index()
    )

    rem_rel = (
        df[df["is_REM_relevant"]]
        .groupby(group_cols, dropna=False)
        .agg(
            mean_delta_REM_REM_relevant=("delta_REM", "mean"),
            median_delta_REM_REM_relevant=("delta_REM", "median"),
            mean_P_REM_EEGonly_REM_relevant=("P_REM_EEGonly", "mean"),
            mean_P_REM_FULL_REM_relevant=("P_REM_FULL", "mean"),
            mean_duration_REM_relevant=("event_duration_sec", "mean"),
            mean_max_EMG_z_REM_relevant=("max_EMG_z_numeric", "mean"),
        )
        .reset_index()
    )

    metrics = metrics.merge(rem_rel, on=group_cols, how="left")

    for col in [
        "REM_relevant",
        "candidate_RBD_auto",
        "stable_REM_EMG",
        "EMG_suppressed_REM",
        "dissociation_positive",
        "transition_REM_Wake",
        "wake_like",
    ]:
        n_col = f"n_{col}"
        if n_col in metrics.columns:
            metrics[f"pct_{col}"] = 100 * metrics[n_col] / metrics["n_events"]

    return metrics


def compute_group_summary(metrics):
    numeric_cols = [
        c for c in metrics.columns
        if c not in ["group", "genotype", "sex", "mouse_id", "week"]
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]

    rows = []
    for (group, week), sub in metrics.groupby(["group", "week"], dropna=False):
        row = {"group": group, "week": week, "n_mice": sub["mouse_id"].nunique()}
        for col in numeric_cols:
            vals = sub[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{col}_sem"] = vals.sem() if len(vals) > 1 else np.nan
            row[f"{col}_median"] = vals.median() if len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["week", "group"])


def run_statistics(metrics, selected_metrics):
    rows = []

    # Between-group tests at each week
    for metric in selected_metrics:
        for week, sub in metrics.groupby("week"):
            wt = sub.loc[sub["group"] == "WT", metric].dropna()
            pdg = sub.loc[sub["group"] == "PD", metric].dropna()

            if len(wt) >= 2 and len(pdg) >= 2:
                stat, p = mannwhitneyu(wt, pdg, alternative="two-sided")
                rows.append({
                    "test_family": "between_group_by_week",
                    "metric": metric,
                    "contrast": f"PD_vs_WT_week_{int(week)}",
                    "week": int(week),
                    "group": "",
                    "n_WT": len(wt),
                    "n_PD": len(pdg),
                    "statistic": stat,
                    "p_value": p,
                    "note": "Mann-Whitney U on mouse/week values",
                })
            else:
                rows.append({
                    "test_family": "between_group_by_week",
                    "metric": metric,
                    "contrast": f"PD_vs_WT_week_{int(week)}",
                    "week": int(week),
                    "group": "",
                    "n_WT": len(wt),
                    "n_PD": len(pdg),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "Skipped: fewer than 2 mice in at least one group",
                })

    # Within-group paired tests vs week 2
    for metric in selected_metrics:
        for group, gsub in metrics.groupby("group"):
            wide = gsub.pivot_table(index="mouse_id", columns="week", values=metric, aggfunc="mean")
            if 2 not in wide.columns:
                continue

            for week in sorted([w for w in wide.columns if w != 2]):
                paired = wide[[2, week]].dropna()
                if len(paired) >= 3:
                    stat, p = wilcoxon(paired[2], paired[week])
                    rows.append({
                        "test_family": "within_group_vs_week2",
                        "metric": metric,
                        "contrast": f"{group}_week_{int(week)}_vs_week_2",
                        "week": int(week),
                        "group": group,
                        "n_pairs": len(paired),
                        "statistic": stat,
                        "p_value": p,
                        "note": "Paired Wilcoxon on mice present at both time points",
                    })
                else:
                    rows.append({
                        "test_family": "within_group_vs_week2",
                        "metric": metric,
                        "contrast": f"{group}_week_{int(week)}_vs_week_2",
                        "week": int(week),
                        "group": group,
                        "n_pairs": len(paired),
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "note": "Skipped: fewer than 3 paired mice",
                    })

    # Mixed-effects model: metric ~ group * week + random intercept mouse_id
    if HAS_STATSMODELS:
        for metric in selected_metrics:
            sub = metrics[["mouse_id", "group", "week", metric]].dropna().copy()
            sub["week_numeric"] = pd.to_numeric(sub["week"], errors="coerce")
            sub["group"] = sub["group"].astype(str)

            if len(sub) >= 8 and sub["group"].nunique() == 2 and sub["mouse_id"].nunique() >= 4:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = smf.mixedlm(
                            f"{metric} ~ C(group) * week_numeric",
                            data=sub,
                            groups=sub["mouse_id"],
                        )
                        fit = model.fit(reml=False, method="lbfgs")

                    for term, p in fit.pvalues.items():
                        rows.append({
                            "test_family": "mixed_effects_progression",
                            "metric": metric,
                            "contrast": term,
                            "week": "",
                            "group": "",
                            "n_observations": len(sub),
                            "n_mice": sub["mouse_id"].nunique(),
                            "statistic": fit.params.get(term, np.nan),
                            "p_value": p,
                            "note": "MixedLM: metric ~ group * week_numeric + random intercept for mouse",
                        })
                except Exception as exc:
                    rows.append({
                        "test_family": "mixed_effects_progression",
                        "metric": metric,
                        "contrast": "model_failed",
                        "week": "",
                        "group": "",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "note": repr(exc),
                    })

    stats = pd.DataFrame(rows)

    if len(stats) and "p_value" in stats.columns:
        stats["p_FDR_all"] = bh_fdr(stats["p_value"].values)
        for family, idx in stats.groupby("test_family").groups.items():
            stats.loc[idx, "p_FDR_within_family"] = bh_fdr(stats.loc[idx, "p_value"].values)

    return stats


def plot_metric(metrics, metric, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    weeks = sorted(metrics["week"].dropna().astype(int).unique())

    # individual mouse trajectories
    for (group, mouse), sub in metrics.groupby(["group", "mouse_id"]):
        sub = sub.sort_values("week")
        ax.plot(
            sub["week"],
            sub[metric],
            marker="o",
            linewidth=1,
            alpha=0.35,
        )

    # group means
    for group, sub in metrics.groupby("group"):
        summary = (
            sub.groupby("week")[metric]
            .agg(["mean", "sem", "count"])
            .reset_index()
            .sort_values("week")
        )
        ax.errorbar(
            summary["week"],
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            linewidth=2.5,
            capsize=3,
            label=f"{group} mean",
        )

    ax.set_xticks(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " "))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png = out_dir / f"{metric}.png"
    svg = out_dir / f"{metric}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)

    return png, svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    events = add_metadata_and_flags(events)

    clean_events_path = args.out_dir / "all_weeks_events_with_metric_flags.csv"
    events.to_csv(clean_events_path, index=False)

    metrics = compute_mouse_week_metrics(events)
    metrics_path = args.out_dir / "all_weeks_mouse_week_event_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = compute_group_summary(metrics)
    summary_path = args.out_dir / "all_weeks_group_week_event_metric_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected_metrics = [
        "n_events",
        "n_REM_relevant",
        "n_candidate_RBD_auto",
        "n_EMG_suppressed_REM",
        "n_stable_REM_EMG",
        "pct_REM_relevant",
        "pct_candidate_RBD_auto",
        "pct_EMG_suppressed_REM",
        "pct_stable_REM_EMG",
        "mean_delta_REM_REM_relevant",
        "mean_max_EMG_z_REM_relevant",
        "mean_duration_REM_relevant",
    ]

    selected_metrics = [m for m in selected_metrics if m in metrics.columns]

    stats = run_statistics(metrics, selected_metrics)
    stats_path = args.out_dir / "all_weeks_event_metric_statistics.csv"
    stats.to_csv(stats_path, index=False)

    for metric in selected_metrics:
        try:
            plot_metric(metrics, metric, fig_dir)
        except Exception as exc:
            print(f"Could not plot {metric}: {exc}")

    notes = args.out_dir / "analysis_notes.txt"
    notes.write_text(
        "All-weeks EMG event-composition metric analysis.\n\n"
        "Important caveat:\n"
        "These metrics are event-count/composition based and are not yet normalized by EEG-only REM time.\n"
        "They are useful for sanity checking, exploratory significance, and figure planning.\n"
        "For paper-level claims, prioritize REM-normalized metrics in the next analysis step.\n\n"
        "Recommended primary future metrics:\n"
        "- candidate RBD-like events per EEG-only REM minute\n"
        "- EMG-suppressed REM events per EEG-only REM minute\n"
        "- stable REM EMG bursts per stable REM minute\n"
        "- REM fragmentation index\n"
        "- percent REM epochs with EMG burst\n",
        encoding="utf-8",
    )

    print("\nWrote outputs to:")
    print(args.out_dir)

    print("\nKey files:")
    print(metrics_path)
    print(summary_path)
    print(stats_path)
    print(fig_dir)

    print("\nMouse/week metrics preview:")
    print(metrics.sort_values(["week", "group", "mouse_id"]).to_string(index=False))

    print("\nGroup/week summary preview:")
    base_cols = ["group", "week", "n_mice"]
    show_cols = [c for c in summary.columns if c in base_cols or c.endswith("_mean")]
    print(summary[show_cols[:20]].sort_values(["week", "group"]).to_string(index=False))


if __name__ == "__main__":
    main()
