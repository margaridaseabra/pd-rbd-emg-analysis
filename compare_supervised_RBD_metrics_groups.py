from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests")

IN = BASE / "supervised_EMG_burst_metrics" / "mouse_level_supervised_RBD_metrics_combined.csv"

OUT_DIR = BASE / "supervised_EMG_burst_metrics" / "group_comparisons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# These are the main metrics to compare.
METRICS = [
    # REM amount and quality
    "total_EEGonly_REM_min",
    "total_stable_EEGonly_REM_min",
    "EEGonly_REM_pct_recording",
    "stable_EEGonly_REM_pct_recording",
    "total_EEGonly_REM_bouts",
    "REM_fragmentation_index_bouts_per_REM_hour",
    "short_REM_bout_fraction_pct",

    # EMG / RBD-like burden
    "total_EMG_events",
    "EMG_events_per_recording_hour",
    "EMG_events_per_EEGonly_REM_min",
    "candidate_EMG_suppressed_REM_per_EEGonly_REM_min",
    "stable_EEG_REM_EMG_burst_per_EEGonly_REM_min",

    # Probability-based metrics
    "mean_EMG_z",
    "mean_max_EMG_z",
    "mean_EEGonly_P_REM",
    "mean_full_P_REM",
    "mean_delta_REM",
    "mean_EEGonly_REM_Wake_balance",
    "mean_EEGonly_ambiguity",
    "mean_full_ambiguity",
    "fraction_near_EEGonly_transition",

    # Event category percentages
    "pct_candidate_EMG_suppressed_REM",
    "pct_stable_EEG_REM_EMG_burst",
    "pct_REM_transition_EMG_burst",
    "pct_mixed_or_ambiguous_state",
    "pct_wake_like_movement",
    "pct_other_EMG_burst",
]


def clean_numeric(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def cliffs_delta(x, y):
    """
    Effect size.
    Positive value means x tends to be larger than y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    lower = 0

    for xi in x:
        greater += np.sum(xi > y)
        lower += np.sum(xi < y)

    return (greater - lower) / (len(x) * len(y))


def bootstrap_mean_ci(x, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    means = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))

    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))

    return np.mean(x), lo, hi


def mann_whitney_test(df, metric, group_a, week_a, group_b, week_b):
    a = clean_numeric(df[(df["group"] == group_a) & (df["week"] == week_a)][metric]).dropna().values
    b = clean_numeric(df[(df["group"] == group_b) & (df["week"] == week_b)][metric]).dropna().values

    if len(a) < 2 or len(b) < 2:
        p = np.nan
        u = np.nan
    else:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        p = res.pvalue
        u = res.statistic

    return {
        "metric": metric,
        "comparison": f"{group_a} W{week_a} vs {group_b} W{week_b}",
        "test": "Mann-Whitney U",
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": np.mean(a) if len(a) else np.nan,
        "mean_b": np.mean(b) if len(b) else np.nan,
        "median_a": np.median(a) if len(a) else np.nan,
        "median_b": np.median(b) if len(b) else np.nan,
        "difference_mean_a_minus_b": (np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan,
        "cliffs_delta_a_vs_b": cliffs_delta(a, b),
        "u_statistic": u,
        "p_value": p,
    }


def paired_week_test(df, metric, group, week_a=2, week_b=21):
    """
    Paired comparison within the same genotype/group using overlapping mouse IDs.
    Only valid if the same mouse IDs truly correspond to the same animals.
    """
    sub = df[df["group"] == group].copy()
    sub[metric] = clean_numeric(sub[metric])

    wide = sub.pivot_table(
        index="mouse_id",
        columns="week",
        values=metric,
        aggfunc="mean",
    )

    if week_a not in wide.columns or week_b not in wide.columns:
        return None

    paired = wide[[week_a, week_b]].dropna()

    if len(paired) < 2:
        p = np.nan
        stat = np.nan
    else:
        res = stats.wilcoxon(paired[week_b], paired[week_a])
        p = res.pvalue
        stat = res.statistic

    diff = paired[week_b] - paired[week_a]

    return {
        "metric": metric,
        "comparison": f"{group} W{week_b} vs {group} W{week_a} paired by mouse_id",
        "test": "Wilcoxon signed-rank",
        "n_pairs": len(paired),
        "mean_week2": paired[week_a].mean() if len(paired) else np.nan,
        "mean_week21": paired[week_b].mean() if len(paired) else np.nan,
        "median_week2": paired[week_a].median() if len(paired) else np.nan,
        "median_week21": paired[week_b].median() if len(paired) else np.nan,
        "mean_change_week21_minus_week2": diff.mean() if len(diff) else np.nan,
        "median_change_week21_minus_week2": diff.median() if len(diff) else np.nan,
        "statistic": stat,
        "p_value": p,
        "paired_mouse_ids": ",".join(map(str, paired.index.tolist())),
    }


def interaction_ols(df, metric):
    """
    Simple exploratory two-factor linear model:
    metric ~ genotype + week + genotype:week

    With small n this is descriptive, not definitive.
    """
    import statsmodels.formula.api as smf

    tmp = df[["group", "week", "mouse_id", metric]].copy()
    tmp[metric] = clean_numeric(tmp[metric])
    tmp = tmp.dropna()

    if len(tmp) < 8 or tmp["group"].nunique() < 2 or tmp["week"].nunique() < 2:
        return {
            "metric": metric,
            "n": len(tmp),
            "interaction_p_value": np.nan,
            "PD_effect_at_week21_estimate": np.nan,
            "model_note": "not enough data",
        }

    tmp["week"] = tmp["week"].astype(int)
    tmp["is_PD"] = (tmp["group"] == "PD").astype(int)
    tmp["is_week21"] = (tmp["week"] == 21).astype(int)

    try:
        model = smf.ols(f"Q('{metric}') ~ is_PD + is_week21 + is_PD:is_week21", data=tmp).fit()

        return {
            "metric": metric,
            "n": len(tmp),
            "interaction_p_value": model.pvalues.get("is_PD:is_week21", np.nan),
            "interaction_estimate": model.params.get("is_PD:is_week21", np.nan),
            "PD_effect_at_week21_estimate": (
                model.params.get("is_PD", np.nan) + model.params.get("is_PD:is_week21", np.nan)
            ),
            "model_r2": model.rsquared,
            "model_note": "exploratory OLS",
        }

    except Exception as e:
        return {
            "metric": metric,
            "n": len(tmp),
            "interaction_p_value": np.nan,
            "interaction_estimate": np.nan,
            "PD_effect_at_week21_estimate": np.nan,
            "model_r2": np.nan,
            "model_note": repr(e),
        }


def plot_metric(df, metric):
    order = [("WT", 2), ("WT", 21), ("PD", 2), ("PD", 21)]
    labels = ["WT W2", "WT W21", "PD W2", "PD W21"]

    data = []
    for group, week in order:
        vals = clean_numeric(df[(df["group"] == group) & (df["week"] == week)][metric]).dropna().values
        data.append(vals)

    if sum(len(v) for v in data) == 0:
        return

    means = [np.mean(v) if len(v) else np.nan for v in data]
    sems = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in data]

    x = np.arange(len(labels))

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(x, means, yerr=sems, capsize=4, alpha=0.65)

    for i, vals in enumerate(data):
        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0]
            plt.scatter(np.full(len(vals), x[i]) + jitter, vals, s=55, zorder=3)

    # Connect paired mouse IDs within each group if present.
    for group, color_x0, color_x1 in [("WT", 0, 1), ("PD", 2, 3)]:
        sub = df[df["group"] == group].copy()
        for mouse_id, sm in sub.groupby("mouse_id"):
            vals = {}
            for week in [2, 21]:
                v = clean_numeric(sm[sm["week"] == week][metric]).dropna()
                if len(v):
                    vals[week] = v.mean()
            if 2 in vals and 21 in vals:
                x0 = 0 if group == "WT" else 2
                x1 = 1 if group == "WT" else 3
                plt.plot([x0, x1], [vals[2], vals[21]], linewidth=1, alpha=0.45)

    plt.xticks(x, labels)
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()

    out = OUT_DIR / f"{metric}.png"
    plt.savefig(out, dpi=180)
    plt.close()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if not IN.exists():
    raise FileNotFoundError(f"Could not find input table: {IN}")

df = pd.read_csv(IN)

df["group"] = df["group"].astype(str)
df["week"] = df["week"].astype(int)

available_metrics = [m for m in METRICS if m in df.columns]

missing = [m for m in METRICS if m not in df.columns]
if missing:
    print("Missing metrics skipped:")
    for m in missing:
        print(" ", m)

print("\nAvailable metrics:")
for m in available_metrics:
    print(" ", m)

# Group summary with mean, median, SEM, bootstrap CI
summary_rows = []

for metric in available_metrics:
    for group, week in [("WT", 2), ("WT", 21), ("PD", 2), ("PD", 21)]:
        vals = clean_numeric(df[(df["group"] == group) & (df["week"] == week)][metric]).dropna().values
        mean, ci_lo, ci_hi = bootstrap_mean_ci(vals)

        summary_rows.append({
            "metric": metric,
            "group": group,
            "week": week,
            "n_mice": len(vals),
            "mean": mean,
            "median": np.median(vals) if len(vals) else np.nan,
            "std": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "sem": np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
            "bootstrap_mean_ci_low": ci_lo,
            "bootstrap_mean_ci_high": ci_hi,
        })

group_summary = pd.DataFrame(summary_rows)
group_summary.to_csv(OUT_DIR / "metric_group_week_summary.csv", index=False)

# Pairwise comparisons
pairwise_rows = []

for metric in available_metrics:
    # Disease effect at week 21
    pairwise_rows.append(mann_whitney_test(df, metric, "PD", 21, "WT", 21))

    # Disease effect at week 2
    pairwise_rows.append(mann_whitney_test(df, metric, "PD", 2, "WT", 2))

    # Time effect within PD
    pairwise_rows.append(mann_whitney_test(df, metric, "PD", 21, "PD", 2))

    # Time effect within WT
    pairwise_rows.append(mann_whitney_test(df, metric, "WT", 21, "WT", 2))

pairwise = pd.DataFrame(pairwise_rows)
pairwise.to_csv(OUT_DIR / "metric_pairwise_mannwhitney_tests.csv", index=False)

# Paired comparisons when same mouse_id exists at W2 and W21.
paired_rows = []
for metric in available_metrics:
    for group in ["WT", "PD"]:
        res = paired_week_test(df, metric, group)
        if res is not None:
            paired_rows.append(res)

paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT_DIR / "metric_paired_week_tests_if_same_mice.csv", index=False)

# Interaction model
interaction_rows = []
for metric in available_metrics:
    interaction_rows.append(interaction_ols(df, metric))

interaction = pd.DataFrame(interaction_rows)
interaction.to_csv(OUT_DIR / "metric_exploratory_genotype_week_interaction.csv", index=False)

# Create plots
PLOT_METRICS = [
    "candidate_EMG_suppressed_REM_per_EEGonly_REM_min",
    "stable_EEG_REM_EMG_burst_per_EEGonly_REM_min",
    "EMG_events_per_EEGonly_REM_min",
    "mean_delta_REM",
    "mean_EEGonly_P_REM",
    "mean_full_P_REM",
    "mean_EEGonly_ambiguity",
    "fraction_near_EEGonly_transition",
    "REM_fragmentation_index_bouts_per_REM_hour",
    "short_REM_bout_fraction_pct",
    "pct_candidate_EMG_suppressed_REM",
    "pct_stable_EEG_REM_EMG_burst",
    "pct_wake_like_movement",
]

for metric in PLOT_METRICS:
    if metric in available_metrics:
        plot_metric(df, metric)

print("\nDone.")
print("Output folder:")
print(OUT_DIR)
print()
print("Main files:")
print(" ", OUT_DIR / "metric_group_week_summary.csv")
print(" ", OUT_DIR / "metric_pairwise_mannwhitney_tests.csv")
print(" ", OUT_DIR / "metric_paired_week_tests_if_same_mice.csv")
print(" ", OUT_DIR / "metric_exploratory_genotype_week_interaction.csv")

print("\nMost important comparison: PD W21 vs WT W21")
pd21_vs_wt21 = pairwise[pairwise["comparison"] == "PD W21 vs WT W21"].copy()
cols = [
    "metric",
    "n_a",
    "n_b",
    "mean_a",
    "mean_b",
    "difference_mean_a_minus_b",
    "cliffs_delta_a_vs_b",
    "p_value",
]
print(pd21_vs_wt21[cols].to_string(index=False))
