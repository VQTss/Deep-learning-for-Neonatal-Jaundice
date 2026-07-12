"""
Histogram of blood(mg/dL) per fold, one figure per fold.

Output:
  figures/blood_hist_per_fold/fold{k}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "datasets/split_10fold_blood.csv"
OUT_DIR = "figures/blood_hist_per_fold"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Patient-level blood (one value per patient)
patient_blood = df.drop_duplicates(subset=["patient_id"])[["patient_id", "blood(mg/dL)"]]

# Common bins across all folds for fair visual comparison
bins = 30
bin_min = float(patient_blood["blood(mg/dL)"].min())
bin_max = float(patient_blood["blood(mg/dL)"].max())
# Pad the upper edge a tiny bit so the max value is included
bin_edges = np.linspace(bin_min, bin_max + 1e-9, bins + 1)

# Same per-fold patient + split role
patient_fold = df.drop_duplicates(subset=["fold", "patient_id"])[
    ["fold", "patient_id", "blood(mg/dL)", "split"]
]

folds = sorted(patient_fold["fold"].unique())

for fold in folds:
    sub = patient_fold[patient_fold["fold"] == fold]
    all_vals = sub["blood(mg/dL)"]
    train_vals = sub.loc[sub["split"] == "train", "blood(mg/dL)"]
    val_vals = sub.loc[sub["split"] == "val", "blood(mg/dL)"]
    test_vals = sub.loc[sub["split"] == "test", "blood(mg/dL)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        [train_vals, val_vals, test_vals],
        bins=bin_edges,
        stacked=True,
        color=["#4C72B0", "#DD8452", "#55A467"],
        label=[
            f"train (n={len(train_vals)})",
            f"val   (n={len(val_vals)})",
            f"test  (n={len(test_vals)})",
        ],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_title(
        f"Fold {fold} — blood(mg/dL) histogram  "
        f"(patients={len(sub)}, all={len(all_vals)})"
    )
    ax.set_xlabel("blood (mg/dL)")
    ax.set_ylabel("number of patients")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", frameon=False)

    # Mean / median reference lines on the full fold distribution
    ax.axvline(all_vals.mean(), color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(all_vals.median(), color="black", linestyle=":", linewidth=1, alpha=0.6)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"fold{int(fold):02d}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")