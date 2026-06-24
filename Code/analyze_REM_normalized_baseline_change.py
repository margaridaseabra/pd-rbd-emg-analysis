#!/usr/bin/env python3
"""
Baseline-change analysis for all-weeks REM-normalized EMG/RBD metrics.

Main question:
    For each mouse, how does each metric change from week 2?
    Is the change from week 2 different in PD vs WT?

This is more appropriate for longitudinal repeated-mouse data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


PRIMARY_METRICS = [
    "candidate_RBD_like_per_EEGonly_REM_prob_min",
    "EMG_suppressed_REM_per_EEGonly_REM_prob_min",
    "REM_relevant_per_EEGonly_REM_prob_min",
    "stable_REM_EMG_per_stable_EEGonly_REM_min",
    "mean_delta_REM_REM_relevant",
    "REM_hc_bouts_per_REM_hc_hour",
    "REM_fraction_prob",
]


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


def cliffs_delta(x, y):
    """
    Cliff's delta.
    Positive means x > y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)

    return total / (len(x) * len(y))


def make_baseline_change_table(df, metrics, baseline_week=2, exclude_weeks=None):
    if exclude_weeks is None:
        exclude_weeks = []

    rows = []

    keep_cols = ["mouse_id", "group", "genotype", "sex", "week"]
    df = df.copy()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    for metric in metrics:
        if metric not in df.columns:
            continue

        sub = df[keep_cols + [metric]].dropna(subset=["mouse_id", "week"]).copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")

        base = (
            sub[sub["week"] == baseline_week]
            .set_index("mouse_id")[metric]
            .rename("baseline_value")
        )

        for _, row in sub.iterrows():
            mouse_id = row["mouse_id"]
            week = int(row["week"])

            if week in exclude_weeks:
                continue

            if mouse_id not in base.index:
                continue

            baseline_value = base.loc[mouse_id]
            value = row[metric]

            if pd.isna(baseline_value) or pd.isna(value):
                continue

            eps = 1e-6

            rows.append({
                "metric": metric,
                "mouse_id": mouse_id,
                "group": row["group"],
                "genotype": row.get("genotype", ""),
                "sex": row.get("sex", ""),
                "week": week,
                "baseline_week": baseline_week,
                "baseline_value": baseline_value,
                "value": value,
                "delta_from_week2": value - baseline_value,
                "log2_ratio_from_week2": np.log2((value + eps) / (baseline_value + eps)),
                "percent_change_from_week2": 100 * (value - baseline_value) / (baseline_value + eps),
            })

    return pd.DataFrame(rows)


def run_baseline_change_stats(changes):
    rows = []

    for metric, msub in changes.groupby("metric"):
        for week, wsub in msub.groupby("week"):
            pd_vals = wsub.loc[wsub["group"] == "PD", "delta_from_week2"].dropna()
            wt_vals = wsub.loc[wsub["group"] == "WT", "delta_from_week2"].dropna()

            if len(pd_vals) >= 2 and len(wt_vals) >= 2:
                stat, p = mannwhitneyu(pd_vals, wt_vals, alternative="two-sided")
                rows.append({
                    "test_family": "PD_vs_WT_change_from_week2",
                    "metric": metric,
                    "week": week,
                    "contrast": f"PD_vs_WT_delta_week{week}_minus_week2",
                    "n_PD": len(pd_vals),
                    "n_WT": len(wt_vals),
                    "PD_mean_delta": pd_vals.mean(),
                    "WT_mean_delta": wt_vals.mean(),
                    "PD_median_delta": pd_vals.median(),
                    "WT_median_delta": wt_vals.median(),
                    "mean_difference_PD_minus_WT": pd_vals.mean() - wt_vals.mean(),
                    "median_difference_PD_minus_WT": pd_vals.median() - wt_vals.median(),
                    "cliffs_delta_PD_vs_WT": cliffs_delta(pd_vals, wt_vals),
                    "statistic": stat,
                    "p_value": p,
                    "note": "Mann-Whitney U comparing mouse-level change from week 2",
                })

            for group, gsub in wsub.groupby("group"):
                vals = gsub["delta_from_week2"].dropna()
                if len(vals) >= 3:
                    stat, p = wilcoxon(vals)
                    rows.append({
                        "test_family": "within_group_change_from_week2",
                        "metric": metric,
                        "week": week,
                        "group": group,
                        "contrast": f"{group}_delta_week{week}_minus_week2_vs_zero",
                        "n_pairs": len(vals),
                        "mean_delta": vals.mean(),
                        "median_delta": vals.median(),
                        "statistic": stat,
                        "p_value": p,
                        "note": "Wilcoxon signed-rank test of change from week 2 vs zero",
                    })

    stats = pd.DataFrame(rows)

    if len(stats):
        stats["p_FDR_all"] = bh_fdr(stats["p_value"].values)

        for fam, idx in stats.groupby("test_family").groups.items():
            stats.loc[idx, "p_FDR_within_family"] = bh_fdr(stats.loc[idx, "p_value"].values)

    return stats


def plot_baseline_change(changes, metric, out_dir):
    sub = changes[changes["metric"] == metric].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    weeks = sorted(sub["week"].dropna().astype(int).unique())

    for (group, mouse_id), msub in sub.groupby(["group", "mouse_id"]):
        msub = msub.sort_values("week")
        ax.plot(
            msub["week"],
            msub["delta_from_week2"],
            marker="o",
            linewidth=1,
            alpha=0.35,
        )

    for group, gsub in sub.groupby("group"):
        summary = (
            gsub.groupby("week")["delta_from_week2"]
            .agg(["mean", "sem"])
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
            label=f"{group} mean change",
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel("Change from week 2")
    ax.set_title(metric.replace("_", " "))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(out_dir / f"baseline_change_{metric}.png", dpi=300)
    fig.savefig(out_dir / f"baseline_change_{metric}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--exclude-week5", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics)

    metrics = [m for m in PRIMARY_METRICS if m in df.columns]

    exclude_weeks = [5] if args.exclude_week5 else []

    changes = make_baseline_change_table(
        df,
        metrics=metrics,
        baseline_week=2,
        exclude_weeks=exclude_weeks,
    )

    changes_path = args.out_dir / "baseline_change_from_week2_by_mouse.csv"
    changes.to_csv(changes_path, index=False)

    stats = run_baseline_change_stats(changes)
    stats_path = args.out_dir / "baseline_change_from_week2_statistics.csv"
    stats.to_csv(stats_path, index=False)

    summary = (
        changes.groupby(["metric", "week", "group"])
        .agg(
            n_mice=("mouse_id", "nunique"),
            mean_delta=("delta_from_week2", "mean"),
            sem_delta=("delta_from_week2", "sem"),
            median_delta=("delta_from_week2", "median"),
            mean_log2_ratio=("log2_ratio_from_week2", "mean"),
            median_log2_ratio=("log2_ratio_from_week2", "median"),
        )
        .reset_index()
        .sort_values(["metric", "week", "group"])
    )

    summary_path = args.out_dir / "baseline_change_from_week2_summary.csv"
    summary.to_csv(summary_path, index=False)

    for metric in metrics:
        plot_baseline_change(changes, metric, fig_dir)

    notes = (
        "Baseline-change analysis from week 2.\n\n"
        "This analysis asks whether each mouse changes relative to its own week 2 baseline.\n"
        "The main between-group test compares PD vs WT change-from-baseline at each later week.\n"
        "Week 5 is excluded if --exclude-week5 is used because it has very low n.\n\n"
        "Positive delta means the metric increased relative to week 2.\n"
    )
    (args.out_dir / "analysis_notes.txt").write_text(notes, encoding="utf-8")

    print("\nWrote:")
    print(changes_path)
    print(summary_path)
    print(stats_path)
    print(fig_dir)

    print("\nTop nominal results:")
    if len(stats):
        show = stats.sort_values("p_value").head(30)
        print(show.to_string(index=False))
    else:
        print("No stats computed.")


if __name__ == "__main__":
    main()
