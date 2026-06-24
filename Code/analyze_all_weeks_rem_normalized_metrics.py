#!/usr/bin/env python3
"""
REM-normalized all-weeks EMG/RBD-like metric analysis.

Primary idea:
    event counts are normalized by EEG-only REM opportunity.

Outputs:
    - REM opportunity table per mouse/week
    - mouse/week REM-normalized metrics
    - group/week summary
    - exploratory statistics
    - longitudinal figures

Important:
    Statistics are done at mouse/week level, not event level.
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


EPOCH_SEC_DEFAULT = 5.0

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

REM_RELEVANT = {
    "stable_REM_EMG_burst",
    "stable_EEG_REM_EMG_burst",
    "EMG_suppressed_REM",
    "candidate_EMG_suppressed_REM",
    "mixed_REM_Wake_transition",
    "REM_transition_EMG_burst",
}

STABLE_REM = {
    "stable_REM_EMG_burst",
    "stable_EEG_REM_EMG_burst",
}

EMG_SUPPRESSED_REM = {
    "EMG_suppressed_REM",
    "candidate_EMG_suppressed_REM",
}


def normalize_state_name(x: str) -> str:
    x = str(x).strip()
    mapping = {
        "Wake": "Awake",
        "wake": "Awake",
        "W": "Awake",
        "WK": "Awake",
        "AWAKE": "Awake",
        "Awake": "Awake",
        "NREM": "NREM",
        "Nrem": "NREM",
        "SWS": "NREM",
        "REM": "REM",
        "Rem": "REM",
        "PS": "REM",
    }
    return mapping.get(x, x)


def patch_mouse_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mouse_id"] = pd.to_numeric(df["mouse_id"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    df["sex"] = df["mouse_id"].map(
        lambda x: MOUSE_INFO[int(x)]["sex"] if pd.notna(x) and int(x) in MOUSE_INFO else ""
    )
    df["genotype"] = df["mouse_id"].map(
        lambda x: MOUSE_INFO[int(x)]["genotype"] if pd.notna(x) and int(x) in MOUSE_INFO else ""
    )
    df["group"] = df["mouse_id"].map(
        lambda x: MOUSE_INFO[int(x)]["group"] if pd.notna(x) and int(x) in MOUSE_INFO else ""
    )
    return df


def find_probability_column(df: pd.DataFrame) -> str:
    preferred = [
        "file_path_state_probabilities",
        "state_probabilities_path",
        "probability_path",
        "eegonly_prob_path",
        "file_path_probabilities",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = [
        c for c in df.columns
        if "prob" in c.lower() and ("path" in c.lower() or "file" in c.lower())
    ]
    if not candidates:
        raise ValueError(
            "Could not find probability path column. Columns are:\n"
            + "\n".join(df.columns)
        )
    return candidates[0]


def load_probabilities(path: str | Path):
    path = Path(str(path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in [".npz", ".npy"]:
        z = np.load(path, allow_pickle=True)

        if hasattr(z, "files"):
            states = [normalize_state_name(s) for s in list(z.files)]
            probs = np.vstack([np.asarray(z[s], dtype=float) for s in list(z.files)]).T
            return probs, states

        arr = np.asarray(z, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unsupported npy probability shape for {path}: {arr.shape}")

        states = ["Awake", "NREM", "REM"][: arr.shape[1]]
        return arr, states

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = list(df.columns)
        rem_cols = [c for c in cols if "rem" in c.lower()]
        wake_cols = [c for c in cols if "wake" in c.lower() or "awake" in c.lower()]
        nrem_cols = [c for c in cols if "nrem" in c.lower() or "sws" in c.lower()]
        ordered_cols = []

        if wake_cols:
            ordered_cols.append(wake_cols[0])
        if nrem_cols:
            ordered_cols.append(nrem_cols[0])
        if rem_cols:
            ordered_cols.append(rem_cols[0])

        if len(ordered_cols) < 3:
            raise ValueError(f"Could not infer Awake/NREM/REM columns from {path}")

        probs = df[ordered_cols].to_numpy(dtype=float)
        states = ["Awake", "NREM", "REM"]
        return probs, states

    raise ValueError(f"Unsupported probability file type: {path}")


def get_prob(probs: np.ndarray, states: list[str], state: str) -> np.ndarray:
    if state not in states:
        return np.zeros(probs.shape[0], dtype=float)
    return probs[:, states.index(state)]


def count_bouts(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return 0
    starts = mask & np.r_[True, ~mask[:-1]]
    return int(starts.sum())


def opportunity_from_manifests(manifest_paths: list[Path], epoch_sec: float) -> pd.DataFrame:
    rows = []

    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        man = pd.read_csv(manifest_path)
        prob_col = find_probability_column(man)

        man = patch_mouse_metadata(man)

        print(f"\nReading manifest: {manifest_path}")
        print(f"Rows: {len(man)}")
        print(f"Probability column: {prob_col}")

        for i, row in man.iterrows():
            prob_path = row.get(prob_col, "")
            if not isinstance(prob_path, str) or not prob_path.strip():
                continue

            try:
                probs, states = load_probabilities(prob_path)
            except Exception as exc:
                rows.append({
                    "manifest_path": str(manifest_path),
                    "probability_path": str(prob_path),
                    "mouse_id": row.get("mouse_id", np.nan),
                    "week": row.get("week", np.nan),
                    "group": row.get("group", ""),
                    "genotype": row.get("genotype", ""),
                    "sex": row.get("sex", ""),
                    "segment_id": row.get("segment_id", np.nan),
                    "recording_name": row.get("recording_name", ""),
                    "load_error": repr(exc),
                })
                continue

            p_rem = get_prob(probs, states, "REM")
            p_wake = get_prob(probs, states, "Awake")
            p_nrem = get_prob(probs, states, "NREM")

            pred_idx = np.nanargmax(probs, axis=1)
            pred_states = np.array(states)[pred_idx]

            rem_hc = p_rem >= 0.70
            rem_soft = p_rem >= 0.50
            stable_rem = (p_rem >= 0.70) & (p_wake <= 0.30)

            rows.append({
                "manifest_path": str(manifest_path),
                "probability_path": str(prob_path),
                "mouse_id": row.get("mouse_id", np.nan),
                "week": row.get("week", np.nan),
                "group": row.get("group", ""),
                "genotype": row.get("genotype", ""),
                "sex": row.get("sex", ""),
                "segment_id": row.get("segment_id", np.nan),
                "recording_name": row.get("recording_name", ""),
                "n_epochs": len(p_rem),
                "total_minutes": len(p_rem) * epoch_sec / 60.0,

                # Primary denominator: probability-weighted REM time
                "eegonly_REM_prob_minutes": float(np.nansum(p_rem) * epoch_sec / 60.0),

                # Sensitivity denominators
                "eegonly_REM_hc_minutes": float(np.sum(rem_hc) * epoch_sec / 60.0),
                "eegonly_REM_soft_minutes": float(np.sum(rem_soft) * epoch_sec / 60.0),
                "stable_EEGonly_REM_minutes": float(np.sum(stable_rem) * epoch_sec / 60.0),
                "predicted_REM_minutes": float(np.sum(pred_states == "REM") * epoch_sec / 60.0),

                # Sleep opportunity sanity checks
                "eegonly_Wake_prob_minutes": float(np.nansum(p_wake) * epoch_sec / 60.0),
                "eegonly_NREM_prob_minutes": float(np.nansum(p_nrem) * epoch_sec / 60.0),

                # Fragmentation
                "REM_hc_bout_count": count_bouts(rem_hc),
                "REM_soft_bout_count": count_bouts(rem_soft),
                "stable_REM_bout_count": count_bouts(stable_rem),
                "load_error": "",
            })

    seg = pd.DataFrame(rows)

    if len(seg) == 0:
        raise RuntimeError("No REM opportunity rows were created. Check manifest/probability paths.")

    ok = seg[seg["load_error"].fillna("").eq("")].copy()

    group_cols = ["group", "genotype", "sex", "mouse_id", "week"]

    agg = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            n_probability_segments=("probability_path", "size"),
            total_minutes=("total_minutes", "sum"),
            eegonly_REM_prob_minutes=("eegonly_REM_prob_minutes", "sum"),
            eegonly_REM_hc_minutes=("eegonly_REM_hc_minutes", "sum"),
            eegonly_REM_soft_minutes=("eegonly_REM_soft_minutes", "sum"),
            stable_EEGonly_REM_minutes=("stable_EEGonly_REM_minutes", "sum"),
            predicted_REM_minutes=("predicted_REM_minutes", "sum"),
            eegonly_Wake_prob_minutes=("eegonly_Wake_prob_minutes", "sum"),
            eegonly_NREM_prob_minutes=("eegonly_NREM_prob_minutes", "sum"),
            REM_hc_bout_count=("REM_hc_bout_count", "sum"),
            REM_soft_bout_count=("REM_soft_bout_count", "sum"),
            stable_REM_bout_count=("stable_REM_bout_count", "sum"),
        )
        .reset_index()
    )

    agg["REM_fraction_prob"] = agg["eegonly_REM_prob_minutes"] / agg["total_minutes"]
    agg["REM_hc_bouts_per_REM_hc_hour"] = agg["REM_hc_bout_count"] / (agg["eegonly_REM_hc_minutes"] / 60.0)
    agg["REM_soft_bouts_per_REM_soft_hour"] = agg["REM_soft_bout_count"] / (agg["eegonly_REM_soft_minutes"] / 60.0)

    return seg, agg


def add_event_flags(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df = patch_mouse_metadata(df)

    if "primary_category" not in df.columns and "event_class" in df.columns:
        df["primary_category"] = df["event_class"]
    elif "primary_category" not in df.columns:
        df["primary_category"] = "other_uncertain"

    for col in ["P_REM_EEGonly", "P_REM_FULL", "P_Wake_EEGonly", "P_Wake_FULL", "delta_REM"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "delta_REM" not in df.columns or df["delta_REM"].isna().all():
        if {"P_REM_EEGonly", "P_REM_FULL"}.issubset(df.columns):
            df["delta_REM"] = df["P_REM_EEGonly"] - df["P_REM_FULL"]
        else:
            df["delta_REM"] = np.nan

    df["is_REM_relevant"] = df["primary_category"].isin(REM_RELEVANT)
    df["is_stable_REM_EMG"] = df["primary_category"].isin(STABLE_REM)
    df["is_EMG_suppressed_REM"] = df["primary_category"].isin(EMG_SUPPRESSED_REM)

    df["is_dissociation_positive"] = (
        (df.get("P_REM_EEGonly", np.nan) >= 0.70)
        & (df.get("P_Wake_EEGonly", np.nan) <= 0.30)
        & (df["delta_REM"] >= 0.25)
    )

    df["is_candidate_RBD_like"] = (
        df["is_stable_REM_EMG"]
        | df["is_EMG_suppressed_REM"]
        | df["is_dissociation_positive"]
    )

    if "duration_sec_for_category" in df.columns:
        df["event_duration_sec"] = pd.to_numeric(df["duration_sec_for_category"], errors="coerce")
    elif "duration_sec" in df.columns:
        df["event_duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    else:
        df["event_duration_sec"] = np.nan

    if "max_EMG_z" in df.columns:
        df["max_EMG_z_numeric"] = pd.to_numeric(df["max_EMG_z"], errors="coerce")
    elif "max_EMG_baseline_z" in df.columns:
        df["max_EMG_z_numeric"] = pd.to_numeric(df["max_EMG_baseline_z"], errors="coerce")
    else:
        df["max_EMG_z_numeric"] = np.nan

    return df


def event_counts_by_mouse_week(events: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["group", "genotype", "sex", "mouse_id", "week"]

    out = (
        events.groupby(group_cols, dropna=False)
        .agg(
            n_events=("primary_category", "size"),
            n_REM_relevant=("is_REM_relevant", "sum"),
            n_candidate_RBD_like=("is_candidate_RBD_like", "sum"),
            n_stable_REM_EMG=("is_stable_REM_EMG", "sum"),
            n_EMG_suppressed_REM=("is_EMG_suppressed_REM", "sum"),
            n_dissociation_positive=("is_dissociation_positive", "sum"),
            mean_delta_REM_all_events=("delta_REM", "mean"),
            median_delta_REM_all_events=("delta_REM", "median"),
            mean_event_duration_sec=("event_duration_sec", "mean"),
            mean_max_EMG_z=("max_EMG_z_numeric", "mean"),
        )
        .reset_index()
    )

    rem = (
        events[events["is_REM_relevant"]]
        .groupby(group_cols, dropna=False)
        .agg(
            mean_delta_REM_REM_relevant=("delta_REM", "mean"),
            median_delta_REM_REM_relevant=("delta_REM", "median"),
            mean_duration_REM_relevant_sec=("event_duration_sec", "mean"),
            mean_max_EMG_z_REM_relevant=("max_EMG_z_numeric", "mean"),
        )
        .reset_index()
    )

    out = out.merge(rem, on=group_cols, how="left")
    return out


def safe_rate(num, denom):
    num = pd.to_numeric(num, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return np.where(denom > 0, num / denom, np.nan)


def make_rem_normalized_metrics(counts: pd.DataFrame, opp: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["group", "genotype", "sex", "mouse_id", "week"]

    df = opp.merge(counts, on=group_cols, how="left")

    count_cols = [
        "n_events",
        "n_REM_relevant",
        "n_candidate_RBD_like",
        "n_stable_REM_EMG",
        "n_EMG_suppressed_REM",
        "n_dissociation_positive",
    ]
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Primary paper-style rates
    df["events_per_EEGonly_REM_prob_min"] = safe_rate(df["n_events"], df["eegonly_REM_prob_minutes"])
    df["REM_relevant_per_EEGonly_REM_prob_min"] = safe_rate(df["n_REM_relevant"], df["eegonly_REM_prob_minutes"])
    df["candidate_RBD_like_per_EEGonly_REM_prob_min"] = safe_rate(df["n_candidate_RBD_like"], df["eegonly_REM_prob_minutes"])
    df["EMG_suppressed_REM_per_EEGonly_REM_prob_min"] = safe_rate(df["n_EMG_suppressed_REM"], df["eegonly_REM_prob_minutes"])
    df["dissociation_positive_per_EEGonly_REM_prob_min"] = safe_rate(df["n_dissociation_positive"], df["eegonly_REM_prob_minutes"])

    # Sensitivity rates using hard REM thresholds
    df["REM_relevant_per_EEGonly_REM_hc_min"] = safe_rate(df["n_REM_relevant"], df["eegonly_REM_hc_minutes"])
    df["candidate_RBD_like_per_EEGonly_REM_hc_min"] = safe_rate(df["n_candidate_RBD_like"], df["eegonly_REM_hc_minutes"])

    # Stable REM bursts should be normalized to stable EEG-only REM minutes
    df["stable_REM_EMG_per_stable_EEGonly_REM_min"] = safe_rate(df["n_stable_REM_EMG"], df["stable_EEGonly_REM_minutes"])

    # General burden
    df["events_per_recording_hour"] = safe_rate(df["n_events"], df["total_minutes"] / 60.0)
    df["REM_relevant_per_recording_hour"] = safe_rate(df["n_REM_relevant"], df["total_minutes"] / 60.0)

    # Composition, still useful but secondary
    df["pct_events_REM_relevant"] = safe_rate(100 * df["n_REM_relevant"], df["n_events"])
    df["pct_events_candidate_RBD_like"] = safe_rate(100 * df["n_candidate_RBD_like"], df["n_events"])
    df["pct_events_EMG_suppressed_REM"] = safe_rate(100 * df["n_EMG_suppressed_REM"], df["n_events"])

    return df


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


def run_stats(metrics: pd.DataFrame, selected_metrics: list[str]) -> pd.DataFrame:
    rows = []

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
                    "n_WT": len(wt),
                    "n_PD": len(pdg),
                    "statistic": stat,
                    "p_value": p,
                    "note": "Mann-Whitney U on mouse/week REM-normalized values",
                })
            else:
                rows.append({
                    "test_family": "between_group_by_week",
                    "metric": metric,
                    "contrast": f"PD_vs_WT_week_{int(week)}",
                    "week": int(week),
                    "n_WT": len(wt),
                    "n_PD": len(pdg),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "Skipped: fewer than 2 mice in at least one group",
                })

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
                        "note": "Paired Wilcoxon on mice present at both weeks",
                    })

    if HAS_STATSMODELS:
        for metric in selected_metrics:
            sub = metrics[["mouse_id", "group", "week", metric]].dropna().copy()
            if len(sub) < 8 or sub["mouse_id"].nunique() < 4 or sub["group"].nunique() < 2:
                continue

            sub["week_numeric"] = pd.to_numeric(sub["week"], errors="coerce")
            sub["log_metric"] = np.log10(pd.to_numeric(sub[metric], errors="coerce") + 1e-6)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(
                        "log_metric ~ C(group) * week_numeric",
                        data=sub,
                        groups=sub["mouse_id"],
                    )
                    fit = model.fit(reml=False, method="lbfgs")

                for term, p in fit.pvalues.items():
                    rows.append({
                        "test_family": "mixed_effects_progression",
                        "metric": metric,
                        "contrast": term,
                        "n_observations": len(sub),
                        "n_mice": sub["mouse_id"].nunique(),
                        "statistic": fit.params.get(term, np.nan),
                        "p_value": p,
                        "note": "Exploratory MixedLM on log10(rate + 1e-6): group * week + random mouse intercept",
                    })
            except Exception as exc:
                rows.append({
                    "test_family": "mixed_effects_progression",
                    "metric": metric,
                    "contrast": "model_failed",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": repr(exc),
                })

    stats = pd.DataFrame(rows)
    if len(stats) and "p_value" in stats.columns:
        stats["p_FDR_all"] = bh_fdr(stats["p_value"].values)
        for fam, idx in stats.groupby("test_family").groups.items():
            stats.loc[idx, "p_FDR_within_family"] = bh_fdr(stats.loc[idx, "p_value"].values)

    return stats


def group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in metrics.columns
        if pd.api.types.is_numeric_dtype(metrics[c])
        and c not in ["mouse_id", "week"]
    ]

    rows = []
    for (group, week), sub in metrics.groupby(["group", "week"], dropna=False):
        row = {
            "group": group,
            "week": week,
            "n_mice": sub["mouse_id"].nunique(),
        }
        for c in metric_cols:
            vals = sub[c].dropna()
            row[f"{c}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{c}_sem"] = vals.sem() if len(vals) > 1 else np.nan
            row[f"{c}_median"] = vals.median() if len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["week", "group"])


def plot_metric(metrics: pd.DataFrame, metric: str, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    weeks = sorted(metrics["week"].dropna().astype(int).unique())

    for (group, mouse_id), sub in metrics.groupby(["group", "mouse_id"]):
        sub = sub.sort_values("week")
        ax.plot(sub["week"], sub[metric], marker="o", linewidth=1, alpha=0.35)

    for group, sub in metrics.groupby("group"):
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

    ax.set_xticks(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " "))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(fig_dir / f"{metric}.png", dpi=300)
    fig.savefig(fig_dir / f"{metric}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--eegonly-manifests", required=True, nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epoch-sec", type=float, default=EPOCH_SEC_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events, low_memory=False)
    events = add_event_flags(events)
    events.to_csv(args.out_dir / "events_with_REM_metric_flags.csv", index=False)

    segment_opp, mouse_opp = opportunity_from_manifests(args.eegonly_manifests, args.epoch_sec)
    segment_opp.to_csv(args.out_dir / "REM_opportunity_by_segment.csv", index=False)
    mouse_opp.to_csv(args.out_dir / "REM_opportunity_by_mouse_week.csv", index=False)

    counts = event_counts_by_mouse_week(events)
    counts.to_csv(args.out_dir / "event_counts_by_mouse_week.csv", index=False)

    metrics = make_rem_normalized_metrics(counts, mouse_opp)
    metrics_path = args.out_dir / "mouse_week_REM_normalized_EMG_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = group_summary(metrics)
    summary_path = args.out_dir / "group_week_REM_normalized_EMG_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = [
        "REM_relevant_per_EEGonly_REM_prob_min",
        "candidate_RBD_like_per_EEGonly_REM_prob_min",
        "EMG_suppressed_REM_per_EEGonly_REM_prob_min",
        "stable_REM_EMG_per_stable_EEGonly_REM_min",
        "dissociation_positive_per_EEGonly_REM_prob_min",
        "REM_hc_bouts_per_REM_hc_hour",
        "REM_soft_bouts_per_REM_soft_hour",
        "REM_fraction_prob",
        "mean_delta_REM_REM_relevant",
        "mean_max_EMG_z_REM_relevant",
    ]
    selected = [m for m in selected if m in metrics.columns]

    stats = run_stats(metrics, selected)
    stats_path = args.out_dir / "REM_normalized_metric_statistics.csv"
    stats.to_csv(stats_path, index=False)

    for m in selected:
        try:
            plot_metric(metrics, m, fig_dir)
        except Exception as exc:
            print(f"Could not plot {m}: {exc}")

    notes = (
        "REM-normalized all-weeks EMG/RBD-like metrics.\n\n"
        "Primary denominator: EEG-only REM probability-weighted minutes.\n"
        "This is preferred because the full EEG+EMG model may suppress REM probability during EMG bursts.\n\n"
        "Primary metrics:\n"
        "- REM_relevant_per_EEGonly_REM_prob_min\n"
        "- candidate_RBD_like_per_EEGonly_REM_prob_min\n"
        "- EMG_suppressed_REM_per_EEGonly_REM_prob_min\n"
        "- stable_REM_EMG_per_stable_EEGonly_REM_min\n"
        "- dissociation_positive_per_EEGonly_REM_prob_min\n\n"
        "Statistics are exploratory and performed on mouse/week values, not event rows.\n"
        "Week 5 may be underpowered if only one WT and one PD mouse were successfully processed.\n"
    )
    (args.out_dir / "analysis_notes.txt").write_text(notes, encoding="utf-8")

    print("\nWrote:")
    print(metrics_path)
    print(summary_path)
    print(stats_path)
    print(fig_dir)

    print("\nREM opportunity preview:")
    print(mouse_opp.sort_values(["week", "group", "mouse_id"]).to_string(index=False))

    print("\nREM-normalized metrics preview:")
    preview_cols = [
        "group", "mouse_id", "week",
        "eegonly_REM_prob_minutes",
        "n_REM_relevant",
        "REM_relevant_per_EEGonly_REM_prob_min",
        "n_candidate_RBD_like",
        "candidate_RBD_like_per_EEGonly_REM_prob_min",
        "n_EMG_suppressed_REM",
        "EMG_suppressed_REM_per_EEGonly_REM_prob_min",
    ]
    preview_cols = [c for c in preview_cols if c in metrics.columns]
    print(metrics[preview_cols].sort_values(["week", "group", "mouse_id"]).to_string(index=False))

    print("\nGroup/week summary saved here:")
    print(summary_path)


if __name__ == "__main__":
    main()
