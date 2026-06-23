from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_DIR = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/somnotate_transition_summary")
OUT_DIR = SUMMARY_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

segment = pd.read_csv(SUMMARY_DIR / "segment_state_probability_summary.csv")
rem = pd.read_csv(SUMMARY_DIR / "rem_bout_summary.csv")
transition = pd.read_csv(SUMMARY_DIR / "state_transitions.csv")

def add_condition(df):
    df = df.copy()
    df["condition"] = df["group"].astype(str) + " week " + df["week"].astype(str)
    return df

segment = add_condition(segment)
rem = add_condition(rem)

condition_order = ["WT week 2", "PD week 2", "WT week 21", "PD week 21"]

def grouped_bar_with_points(df, value_col, ylabel, title, out_name):
    data = df[df["condition"].isin(condition_order)].copy()

    means = data.groupby("condition")[value_col].mean().reindex(condition_order)
    sems = data.groupby("condition")[value_col].sem().reindex(condition_order)

    x = np.arange(len(condition_order))

    plt.figure(figsize=(8, 5))
    plt.bar(x, means.values, yerr=sems.values, capsize=4, alpha=0.7)

    for i, cond in enumerate(condition_order):
        vals = data.loc[data["condition"] == cond, value_col].dropna().values
        if len(vals):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            plt.scatter(np.full(len(vals), i) + jitter, vals, s=25)

    plt.xticks(x, condition_order, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out = OUT_DIR / out_name
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

# 1. State occupancy
for state in ["Awake", "NREM", "REM"]:
    col = f"pct_{state}"
    if col in segment.columns:
        grouped_bar_with_points(
            segment,
            col,
            f"% {state}",
            f"Predicted {state} occupancy by group/week",
            f"occupancy_{state}.png",
        )

# 2. Transitions per hour
grouped_bar_with_points(
    segment,
    "transitions_per_hour",
    "Transitions per hour",
    "Predicted state transitions per hour",
    "transitions_per_hour.png",
)

# 3. Mean confidence
grouped_bar_with_points(
    segment,
    "mean_confidence",
    "Mean max state probability",
    "Mean Somnotate confidence by group/week",
    "mean_confidence.png",
)

# 4. REM summaries
for col, ylabel, title, out in [
    ("total_rem_min", "Total REM minutes", "Total predicted REM time", "rem_total_minutes.png"),
    ("n_rem_bouts", "Number of REM bouts", "Predicted REM bout count", "rem_bout_count.png"),
    ("mean_rem_bout_min", "Mean REM bout duration (min)", "Predicted REM bout duration", "rem_mean_bout_duration.png"),
    ("short_rem_bouts_lt_30s", "REM bouts < 30 sec", "Short predicted REM bouts", "rem_short_bouts.png"),
]:
    if col in rem.columns:
        grouped_bar_with_points(rem, col, ylabel, title, out)

# 5. Transition matrix heatmap
matrix = pd.crosstab(transition["from_state"], transition["to_state"])
states = sorted(set(transition["from_state"]).union(set(transition["to_state"])))
matrix = matrix.reindex(index=states, columns=states, fill_value=0)

plt.figure(figsize=(6, 5))
plt.imshow(matrix.values)
plt.xticks(np.arange(len(states)), states)
plt.yticks(np.arange(len(states)), states)
plt.xlabel("To state")
plt.ylabel("From state")
plt.title("Automated state transition counts")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, str(matrix.values[i, j]), ha="center", va="center")

plt.colorbar(label="Count")
plt.tight_layout()
out = OUT_DIR / "transition_matrix_heatmap.png"
plt.savefig(out, dpi=180)
plt.close()
print("Wrote:", out)

# Save condition-level summary tables
segment_group_summary = segment.groupby("condition", as_index=False).agg(
    n_segments=("segment_id", "count"),
    mean_pct_Awake=("pct_Awake", "mean"),
    mean_pct_NREM=("pct_NREM", "mean"),
    mean_pct_REM=("pct_REM", "mean"),
    mean_confidence=("mean_confidence", "mean"),
    mean_transitions_per_hour=("transitions_per_hour", "mean"),
)

rem_group_summary = rem.groupby("condition", as_index=False).agg(
    n_segments=("segment_id", "count"),
    mean_total_rem_min=("total_rem_min", "mean"),
    mean_n_rem_bouts=("n_rem_bouts", "mean"),
    mean_rem_bout_min=("mean_rem_bout_min", "mean"),
    mean_short_rem_bouts_lt_30s=("short_rem_bouts_lt_30s", "mean"),
)

segment_group_summary.to_csv(OUT_DIR / "segment_group_summary.csv", index=False)
rem_group_summary.to_csv(OUT_DIR / "rem_group_summary.csv", index=False)

print("\nWrote group summaries:")
print(OUT_DIR / "segment_group_summary.csv")
print(OUT_DIR / "rem_group_summary.csv")
print("\nOpen folder:")
print(OUT_DIR)
