#!/usr/bin/env python3
"""
QC-confirmed EMG/RBD-like metrics.

This script uses manually reviewed QC labels to compute:
- QC coverage
- QC-confirmed possible RBD-like events per EEG-only REM minute
- QC label composition by group/week/category
- mouse/week and group/week summaries

Important:
Unreviewed events are not treated as negative. They are unknown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


POSITIVE_LABELS = {
    "possible_RBD_like",
    "probable_RBD_like",
    "confirmed_RBD_like",
    "RBD_like",
}

NEGATIVE_LABELS = {
    "real_burst_but_wake",
    "transition_event",
    "artifact",
    "exclude",
}

UNCERTAIN_LABELS = {
    "uncertain",
}

NOT_REVIEWED_LABELS = {
    "",
    "nan",
    "None",
    "not_reviewed",
    "not reviewed",
}


def find_qc_label_col(df: pd.DataFrame) -> str:
    preferred = [
        "qc_reviewer_label",
        "QC_reviewer_label",
        "reviewer_label",
        "manual_qc_label",
        "qc_label",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = [
        c for c in df.columns
        if "label" in c.lower() and ("qc" in c.lower() or "review" in c.lower())
    ]

    if candidates:
        return candidates[0]

    raise ValueError(
        "Could not find QC label column. Available columns are:\n"
        + "\n".join(df.columns)
    )


def normalize_label(x):
    if pd.isna(x):
        return "not_reviewed"
    x = str(x).strip()
    if x in NOT_REVIEWED_LABELS:
        return "not_reviewed"
    return x


def safe_rate(num, denom):
    num = pd.to_numeric(num, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return np.where(denom > 0, num / denom, np.nan)


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
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    temp = np.empty_like(adj)
    temp[order] = adj
    out[valid] = temp
    return out


def plot_metric(df, metric, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (group, mouse), sub in df.groupby(["group", "mouse_id"]):
        sub = sub.sort_values("week")
        ax.plot(sub["week"], sub[metric], marker="o", linewidth=1, alpha=0.35)

    for group, sub in df.groupby("group"):
        s = (
            sub.groupby("week")[metric]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("week")
        )
        ax.errorbar(
            s["week"],
            s["mean"],
            yerr=s["sem"],
            marker="o",
            linewidth=2.5,
            capsize=3,
            label=f"{group} mean",
        )

    ax.set_xlabel("Week")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " "))
    ax.axhline(0, linewidth=1)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(out_dir / f"{metric}.png", dpi=300)
    fig.savefig(out_dir / f"{metric}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--rem-opportunity", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events, low_memory=False)
    opp = pd.read_csv(args.rem_opportunity, low_memory=False)

    label_col = find_qc_label_col(events)
    events["qc_label_normalized"] = events[label_col].map(normalize_label)

    events["is_reviewed"] = events["qc_label_normalized"].ne("not_reviewed")
    events["is_qc_positive_RBD_like"] = events["qc_label_normalized"].isin(POSITIVE_LABELS)
    events["is_qc_negative"] = events["qc_label_normalized"].isin(NEGATIVE_LABELS)
    events["is_qc_uncertain"] = events["qc_label_normalized"].isin(UNCERTAIN_LABELS)

    group_cols = ["group", "genotype", "sex", "mouse_id", "week"]

    coverage = (
        events.groupby(["group", "week", "primary_category"], dropna=False)
        .agg(
            n_events=("qc_label_normalized", "size"),
            n_reviewed=("is_reviewed", "sum"),
            n_qc_positive_RBD_like=("is_qc_positive_RBD_like", "sum"),
            n_qc_negative=("is_qc_negative", "sum"),
            n_qc_uncertain=("is_qc_uncertain", "sum"),
        )
        .reset_index()
    )
    coverage["pct_reviewed"] = safe_rate(100 * coverage["n_reviewed"], coverage["n_events"])
    coverage["pct_positive_among_reviewed"] = safe_rate(
        100 * coverage["n_qc_positive_RBD_like"],
        coverage["n_reviewed"],
    )

    coverage.to_csv(args.out_dir / "qc_coverage_by_group_week_category.csv", index=False)

    counts = (
        events.groupby(group_cols, dropna=False)
        .agg(
            n_events_total=("qc_label_normalized", "size"),
            n_events_reviewed=("is_reviewed", "sum"),
            n_qc_positive_RBD_like=("is_qc_positive_RBD_like", "sum"),
            n_qc_negative=("is_qc_negative", "sum"),
            n_qc_uncertain=("is_qc_uncertain", "sum"),
        )
        .reset_index()
    )

    keys = ["group", "genotype", "sex", "mouse_id", "week"]
    metrics = opp.merge(counts, on=keys, how="left")

    for c in ["n_events_total", "n_events_reviewed", "n_qc_positive_RBD_like", "n_qc_negative", "n_qc_uncertain"]:
        metrics[c] = metrics[c].fillna(0)

    metrics["qc_positive_RBD_like_per_EEGonly_REM_prob_min"] = safe_rate(
        metrics["n_qc_positive_RBD_like"],
        metrics["eegonly_REM_prob_minutes"],
    )

    metrics["qc_reviewed_events_per_EEGonly_REM_prob_min"] = safe_rate(
        metrics["n_events_reviewed"],
        metrics["eegonly_REM_prob_minutes"],
    )

    metrics["qc_positive_fraction_among_reviewed"] = safe_rate(
        metrics["n_qc_positive_RBD_like"],
        metrics["n_events_reviewed"],
    )

    metrics["qc_review_coverage_fraction"] = safe_rate(
        metrics["n_events_reviewed"],
        metrics["n_events_total"],
    )

    metrics.to_csv(args.out_dir / "qc_confirmed_mouse_week_metrics.csv", index=False)

    summary = (
        metrics.groupby(["group", "week"], dropna=False)
        .agg(
            n_mice=("mouse_id", "nunique"),
            mean_qc_positive_rate=("qc_positive_RBD_like_per_EEGonly_REM_prob_min", "mean"),
            sem_qc_positive_rate=("qc_positive_RBD_like_per_EEGonly_REM_prob_min", "sem"),
            median_qc_positive_rate=("qc_positive_RBD_like_per_EEGonly_REM_prob_min", "median"),
            mean_qc_positive_fraction=("qc_positive_fraction_among_reviewed", "mean"),
            mean_qc_review_coverage=("qc_review_coverage_fraction", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(args.out_dir / "qc_confirmed_group_week_summary.csv", index=False)

    rows = []
    metric = "qc_positive_RBD_like_per_EEGonly_REM_prob_min"

    for week, sub in metrics.groupby("week"):
        wt = sub.loc[sub["group"] == "WT", metric].dropna()
        pdg = sub.loc[sub["group"] == "PD", metric].dropna()

        if len(wt) >= 2 and len(pdg) >= 2:
            stat, p = mannwhitneyu(wt, pdg, alternative="two-sided")
            rows.append({
                "metric": metric,
                "contrast": f"PD_vs_WT_week_{int(week)}",
                "week": int(week),
                "n_WT": len(wt),
                "n_PD": len(pdg),
                "statistic": stat,
                "p_value": p,
                "note": "Mann-Whitney U on mouse/week QC-confirmed rates",
            })

    stats = pd.DataFrame(rows)
    if len(stats):
        stats["p_FDR"] = bh_fdr(stats["p_value"].values)

    stats.to_csv(args.out_dir / "qc_confirmed_statistics.csv", index=False)

    plot_metric(metrics, "qc_positive_RBD_like_per_EEGonly_REM_prob_min", fig_dir)
    plot_metric(metrics, "qc_positive_fraction_among_reviewed", fig_dir)
    plot_metric(metrics, "qc_review_coverage_fraction", fig_dir)

    print("QC label column:", label_col)
    print("Wrote outputs to:", args.out_dir)
    print("\nQC label counts:")
    print(events["qc_label_normalized"].value_counts().to_string())
    print("\nGroup/week QC summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
