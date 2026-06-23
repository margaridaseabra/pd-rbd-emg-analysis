from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
MANIFEST = Path("/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/manifests/all_segments_inference_512hz_completed.csv")

OUT_DIR = Path.home() / "Desktop" / "somnotate_transition_moments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 5
WINDOW_MIN = 8                 # total window shown in figure
HALF_WINDOW_EPOCHS = int((WINDOW_MIN * 60 / EPOCH_SEC) / 2)

# ambiguity scoring window around transition center
LOCAL_SCORE_SEC = 60
LOCAL_SCORE_EPOCHS = int(LOCAL_SCORE_SEC / EPOCH_SEC)

STATE_ORDER = ["Awake", "NREM", "REM", "Undefined"]
STATE_TO_Y = {s: i for i, s in enumerate(STATE_ORDER)}

# transition types to prioritize for figures
PRIORITY_TRANSITIONS = [
    "Awake->NREM",
    "NREM->REM",
    "REM->Awake",
    "NREM->Awake",
]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

def normalize_state(s):
    s = str(s).strip()
    mapping = {
        "Wake": "Awake",
        "W": "Awake",
        "AWAKE": "Awake",
        "wake": "Awake",
        "awake": "Awake",
        "NREM": "NREM",
        "Nrem": "NREM",
        "SWS": "NREM",
        "sws": "NREM",
        "NonREM": "NREM",
        "REM": "REM",
        "Rem": "REM",
        "PS": "REM",
        "ps": "REM",
        "Paradoxical Sleep": "REM",
        "Undefined": "Undefined",
        "undefined": "Undefined",
        "ND": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
    }
    return mapping.get(s, s)

def load_probabilities(path):
    z = np.load(path, allow_pickle=True)
    state_names = list(z.files)
    probs = np.vstack([np.asarray(z[s], dtype=float) for s in state_names]).T
    return probs, state_names

def load_stage_duration(path, n_epochs):
    path = Path(path)
    text = path.read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    states = []

    if not lines:
        return np.array(["Undefined"] * n_epochs)

    # Stage-duration format
    if lines[0].startswith("*Duration"):
        prev_end_sec = 0.0

        for line in lines[2:]:
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue

            label = " ".join(parts[:-1])
            end_sec = float(parts[-1])

            start_epoch = int(round(prev_end_sec / EPOCH_SEC))
            end_epoch = int(round(end_sec / EPOCH_SEC))

            states.extend([normalize_state(label)] * max(0, end_epoch - start_epoch))
            prev_end_sec = end_sec

    else:
        states = [normalize_state(x) for x in lines]

    if len(states) < n_epochs:
        states.extend(["Undefined"] * (n_epochs - len(states)))
    elif len(states) > n_epochs:
        states = states[:n_epochs]

    return np.array(states)

def states_to_y(states):
    return np.array([STATE_TO_Y.get(s, STATE_TO_Y["Undefined"]) for s in states])

def transition_candidates_from_segment(row):
    prob_path = Path(row["file_path_state_probabilities"])
    if not prob_path.exists():
        return []

    probs, state_names = load_probabilities(prob_path)
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)
    n_epochs = len(pred)

    manual = None
    manual_path = row.get("file_path_manual_state_annotation", "")
    if isinstance(manual_path, str) and manual_path:
        manual_path = Path(manual_path)
        if manual_path.exists():
            try:
                manual = load_stage_duration(manual_path, n_epochs)
            except Exception:
                manual = None

    cps = np.where(pred[1:] != pred[:-1])[0] + 1
    out = []

    for cp in cps:
        before_state = pred[cp - 1]
        after_state = pred[cp]
        transition_type = f"{before_state}->{after_state}"

        lo = max(0, cp - LOCAL_SCORE_EPOCHS)
        hi = min(n_epochs, cp + LOCAL_SCORE_EPOCHS)

        local_conf = conf[lo:hi]
        local_probs = probs[lo:hi]

        if len(local_conf) == 0:
            continue

        # ambiguity score: higher means more mixed / uncertain
        ambiguity = 1.0 - np.mean(np.max(local_probs, axis=1))
        min_conf = np.min(local_conf)
        mean_conf = np.mean(local_conf)

        manual_before = ""
        manual_after = ""
        local_manual_agreement = np.nan

        if manual is not None:
            manual_before = manual[max(0, cp - 1)]
            manual_after = manual[min(n_epochs - 1, cp)]
            valid = np.isin(manual[lo:hi], ["Awake", "NREM", "REM"])
            if np.any(valid):
                local_manual_agreement = np.mean(manual[lo:hi][valid] == pred[lo:hi][valid]) * 100

        out.append({
            "recording_name": row["recording_name"],
            "group": row.get("group", ""),
            "week": row.get("week", ""),
            "mouse_id": row["mouse_id"],
            "segment_id": row["segment_id"],
            "transition_epoch": int(cp),
            "transition_min_from_start": cp * EPOCH_SEC / 60,
            "transition_type": transition_type,
            "before_state": before_state,
            "after_state": after_state,
            "ambiguity_score": ambiguity,
            "min_conf_local": min_conf,
            "mean_conf_local": mean_conf,
            "local_manual_agreement_pct": local_manual_agreement,
            "manual_before": manual_before,
            "manual_after": manual_after,
            "file_path_state_probabilities": str(prob_path),
            "file_path_manual_state_annotation": str(manual_path) if manual is not None else "",
        })

    return out

def make_transition_plot(row, candidate):
    prob_path = Path(row["file_path_state_probabilities"])
    probs, state_names = load_probabilities(prob_path)
    pred = np.array(state_names)[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)

    n_epochs = len(pred)
    cp = int(candidate["transition_epoch"])

    start = max(0, cp - HALF_WINDOW_EPOCHS)
    end = min(n_epochs, cp + HALF_WINDOW_EPOCHS)

    rel_t_min = (np.arange(start, end) - cp) * EPOCH_SEC / 60.0

    manual = None
    manual_path = Path(row["file_path_manual_state_annotation"]) if "file_path_manual_state_annotation" in row else None
    if manual_path is not None and manual_path.exists():
        try:
            manual = load_stage_duration(manual_path, n_epochs)
        except Exception:
            manual = None

    n_panels = 4 if manual is not None else 3
    height_ratios = [1, 1, 2, 1] if manual is not None else [1, 2, 1]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios}
    )

    ax_i = 0

    if manual is not None:
        axes[ax_i].step(rel_t_min, states_to_y(manual[start:end]), where="post", linewidth=1.2)
        axes[ax_i].set_yticks(range(len(STATE_ORDER)))
        axes[ax_i].set_yticklabels(STATE_ORDER)
        axes[ax_i].set_ylabel("Manual")
        axes[ax_i].axvline(0, linestyle="--", linewidth=1)
        ax_i += 1

    axes[ax_i].step(rel_t_min, states_to_y(pred[start:end]), where="post", linewidth=1.2)
    axes[ax_i].set_yticks(range(len(STATE_ORDER)))
    axes[ax_i].set_yticklabels(STATE_ORDER)
    axes[ax_i].set_ylabel("Somnotate")
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)
    ax_i += 1

    for j, state in enumerate(state_names):
        axes[ax_i].plot(rel_t_min, probs[start:end, j], label=state, linewidth=1.5)
    axes[ax_i].set_ylim(-0.02, 1.02)
    axes[ax_i].set_ylabel("State probability")
    axes[ax_i].legend(loc="upper right")
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)
    ax_i += 1

    axes[ax_i].plot(rel_t_min, conf[start:end], linewidth=1.2)
    axes[ax_i].set_ylim(-0.02, 1.02)
    axes[ax_i].set_ylabel("Max prob")
    axes[ax_i].set_xlabel("Minutes relative to transition center")
    axes[ax_i].axvline(0, linestyle="--", linewidth=1)

    fig.suptitle(
        f"{candidate['transition_type']} | mouse {row['mouse_id']} | "
        f"week {row.get('week', '')} | segment {row['segment_id']} | "
        f"ambiguity={candidate['ambiguity_score']:.3f} | "
        f"min_conf={candidate['min_conf_local']:.3f}\n"
        f"{row['recording_name']}",
        y=0.98
    )

    plt.tight_layout()

    safe_transition = candidate["transition_type"].replace("->", "_to_")
    out = OUT_DIR / (
        f"{safe_transition}_mouse{row['mouse_id']}_week{row.get('week','')}"
        f"_seg{row['segment_id']}_at_{candidate['transition_min_from_start']:.1f}min.png"
    )
    plt.savefig(out, dpi=180)
    plt.close()
    return out

# -------------------------------------------------
# MAIN
# -------------------------------------------------
df = pd.read_csv(MANIFEST)

# Use WT + manual labels + existing state probabilities
manual_exists = df["file_path_manual_state_annotation"].map(lambda p: Path(p).exists() if isinstance(p, str) else False)
prob_exists = df["file_path_state_probabilities"].map(lambda p: Path(p).exists() if isinstance(p, str) else False)

wt = df[
    (df["group"] == "WT")
    & (df["week"].isin([2, 21]))
    & manual_exists
    & prob_exists
].copy()

if "pct_scored" in wt.columns:
    wt = wt[wt["pct_scored"] >= 0.90].copy()

print("Using WT rows:", len(wt))
print(wt[["recording_name", "mouse_id", "week", "segment_id"]].to_string(index=False))

all_candidates = []
for _, row in wt.iterrows():
    try:
        cands = transition_candidates_from_segment(row)
        all_candidates.extend(cands)
    except Exception as e:
        print("FAILED on", row["recording_name"], row["segment_id"], repr(e))

cand_df = pd.DataFrame(all_candidates)

if len(cand_df) == 0:
    raise SystemExit("No transition candidates found.")

cand_df = cand_df.sort_values(
    ["ambiguity_score", "min_conf_local"],
    ascending=[False, True]
).reset_index(drop=True)

cand_out = OUT_DIR / "transition_candidates_summary.csv"
cand_df.to_csv(cand_out, index=False)
print("\nWrote:", cand_out)

# Choose best example for each priority transition type
chosen_rows = []

used_segment_keys = set()
for ttype in PRIORITY_TRANSITIONS:
    sub = cand_df[cand_df["transition_type"] == ttype].copy()
    if len(sub) == 0:
        continue

    for _, c in sub.iterrows():
        key = (c["mouse_id"], c["week"], c["segment_id"])
        if key not in used_segment_keys:
            chosen_rows.append(c)
            used_segment_keys.add(key)
            break

# Add one extra "most ambiguous overall" example if not already included
if len(cand_df) > 0:
    for _, c in cand_df.iterrows():
        key = (c["mouse_id"], c["week"], c["segment_id"], c["transition_epoch"])
        already = any(
            (x["mouse_id"], x["week"], x["segment_id"], x["transition_epoch"]) == key
            for x in chosen_rows
        )
        if not already:
            chosen_rows.append(c)
            break

chosen_df = pd.DataFrame(chosen_rows)
chosen_out = OUT_DIR / "chosen_transition_examples.csv"
chosen_df.to_csv(chosen_out, index=False)
print("Wrote:", chosen_out)

print("\nCreating figures...")
for _, c in chosen_df.iterrows():
    match = wt[
        (wt["mouse_id"] == c["mouse_id"]) &
        (wt["week"] == c["week"]) &
        (wt["segment_id"] == c["segment_id"]) &
        (wt["recording_name"] == c["recording_name"])
    ]
    if len(match) == 0:
        continue
    row = match.iloc[0]
    out = make_transition_plot(row, c)
    print("Wrote:", out)

print("\nTop candidate transitions:")
print(
    cand_df[
        [
            "transition_type", "mouse_id", "week", "segment_id",
            "transition_min_from_start", "ambiguity_score",
            "min_conf_local", "local_manual_agreement_pct"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nDone.")
print("Open folder:")
print(OUT_DIR)
