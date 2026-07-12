"""
Plot per-fold blood(mg/dL) distribution from split_10fold_blood.csv.

CSV layout: every (patient_id, image_idx) appears in 10 rows — one per fold —
with a 'split' column indicating the role (train/val/test) the patient plays
in that fold. To avoid double-counting patients within a fold we deduplicate
on (fold, patient_id) and use the single blood(mg/dL) value per patient.
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "datasets/split_10fold_blood.csv"
OUT_PATH = "figures/blood_distribution_per_fold.png"

df = pd.read_csv(CSV_PATH)

# One row per (fold, patient) — patients have a single blood value across folds
patient_fold = df.drop_duplicates(subset=["fold", "patient_id"])[
    ["fold", "patient_id", "blood(mg/dL)", "split"]
].copy()

folds = sorted(patient_fold["fold"].unique())

# Color per split for visual context (test patients highlighted)
split_colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A467"}

fig, axes = plt.subplots(
    nrows=1, ncols=len(folds), sharey=True, sharex=True, figsize=(20, 10)
)

for ax, fold in zip(axes, folds):
    sub = patient_fold[patient_fold["fold"] == fold]
    # Sort patients by blood to make the strip readable
    sub = sub.sort_values("blood(mg/dL)")
    for sp, color in split_colors.items():
        s = sub[sub["split"] == sp]
        ax.scatter(
            s["blood(mg/dL)"],
            s["patient_id"],
            c=color,
            s=14,
            alpha=0.85,
            label=sp,
            edgecolors="none",
        )
    ax.set_title(f"Fold {fold}")
    ax.set_xlabel("blood (mg/dL)")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("patient_id")

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
)

fig.suptitle(
    "Per-fold blood(mg/dL) distribution per patient — split_10fold_blood.csv",
    y=1.05,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")