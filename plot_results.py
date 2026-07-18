#!/usr/bin/env python3
"""Parse training logs across ROI variants and produce comparison figures."""

import os
import re
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CHECKPOINT_ROOT = Path("checkpoint/ROI_variants")
ROI_SIZES = [96, 128, 160, 224]
FIG_DIR = CHECKPOINT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+\|\s+"
    r"LR_bb:\s+(?P<lr_bb>[\d.eE+\-]+)\s+\|\s+LR_head:\s+(?P<lr_head>[\d.eE+\-]+)\s+\|\s+"
    r"Train Loss:\s+(?P<train_loss>[\d.]+)\s+\|\s+Val Loss:\s+(?P<val_loss>[\d.]+)\s+\|\s+"
    r"Val MAE:\s+(?P<val_mae>[\d.]+)\s+\|\s+Val RMSE:\s+(?P<val_rmse>[\d.]+)\s+\|\s+"
    r"Val R2:\s+(?P<val_r2>[-+eE\d.]+)\s+\|\s+Time:\s+(?P<time>[\d.]+)s"
)

TEST_RE = re.compile(
    r"Test Loss:\s+(?P<test_loss>[\d.]+)\s*$"
)
TEST_MAE_RE = re.compile(r"Test MAE:\s+(?P<mae>[\d.]+)\s*$")
TEST_RMSE_RE = re.compile(r"Test RMSE:\s+(?P<rmse>[\d.]+)\s*$")
TEST_R2_RE = re.compile(r"Test R2:\s+(?P<r2>[-+eE\d.]+)\s*$")

TABLE_RE = re.compile(
    r"^\s*(?P<fold>\d+)\s+(?P<loss>[\d.]+)\s+(?P<mae>[\d.]+)\s+(?P<mse>[\d.]+)\s+"
    r"(?P<rmse>[\d.]+)\s+(?P<r2>[-+eE\d.]+)\s*$"
)


def parse_log(log_path: Path):
    """Return per-fold metrics (epoch-level) and per-fold test metrics."""
    fold_epochs = {}  # fold -> list[dict]
    fold_test = {}    # fold -> dict
    current_fold = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group("epoch"))
                if current_fold not in fold_epochs:
                    fold_epochs[current_fold] = []
                fold_epochs[current_fold].append({
                    "epoch": epoch,
                    "lr_bb": float(m.group("lr_bb")),
                    "lr_head": float(m.group("lr_head")),
                    "train_loss": float(m.group("train_loss")),
                    "val_loss": float(m.group("val_loss")),
                    "val_mae": float(m.group("val_mae")),
                    "val_rmse": float(m.group("val_rmse")),
                    "val_r2": float(m.group("val_r2")),
                    "time": float(m.group("time")),
                })
                continue

            if "Training Fold" in line:
                m2 = re.search(r"Training Fold\s+(\d+)/(\d+)", line)
                if m2:
                    current_fold = int(m2.group(1))
                    fold_epochs.setdefault(current_fold, [])
                continue
            if "Testing Fold" in line:
                m2 = re.search(r"Testing Fold\s+(\d+)/(\d+)", line)
                if m2:
                    current_fold = int(m2.group(1))
                continue

            m3 = TEST_RE.search(line)
            if m3:
                fold_test.setdefault(current_fold, {})["test_loss"] = float(m3.group("test_loss"))
                continue
            m4 = TEST_MAE_RE.search(line)
            if m4:
                fold_test.setdefault(current_fold, {})["mae"] = float(m4.group("mae"))
                continue
            m5 = TEST_RMSE_RE.search(line)
            if m5:
                fold_test.setdefault(current_fold, {})["rmse"] = float(m5.group("rmse"))
                continue
            m6 = TEST_R2_RE.search(line)
            if m6:
                fold_test.setdefault(current_fold, {})["r2"] = float(m6.group("r2"))
                continue

            m7 = TABLE_RE.search(line)
            if m7:
                fold = int(m7.group("fold"))
                fold_test[fold] = {
                    "test_loss": float(m7.group("loss")),
                    "mae": float(m7.group("mae")),
                    "mse": float(m7.group("mse")),
                    "rmse": float(m7.group("rmse")),
                    "r2": float(m7.group("r2")),
                }
                continue

    fold_epochs.pop(None, None)
    fold_test = {k: v for k, v in fold_test.items() if k is not None}
    return fold_epochs, fold_test


def aggregate_test(fold_test):
    keys = ["test_loss", "mae", "rmse", "r2"]
    arr = {k: np.array([v[k] for v in fold_test.values() if k in v]) for k in keys}
    return {k: (float(arr[k].mean()), float(arr[k].std())) for k in keys}


def main():
    palette = {
        96:  "#1f77b4",
        128: "#2ca02c",
        160: "#ff7f0e",
        224: "#d62728",
    }

    data = {}
    for roi in ROI_SIZES:
        log_path = CHECKPOINT_ROOT / f"roi{roi}" / f"roi{roi}" / "logs"
        logs = sorted(log_path.glob("train_*.log"))
        if not logs:
            print(f"[WARN] No log for ROI {roi}")
            continue
        log_path = logs[-1]
        print(f"[INFO] Parsing {log_path}")
        fold_epochs, fold_test = parse_log(log_path)
        agg = aggregate_test(fold_test) if fold_test else None
        data[roi] = {
            "log_path": log_path,
            "fold_epochs": fold_epochs,
            "fold_test": fold_test,
            "agg": agg,
        }
        print(f"  folds: {len(fold_epochs)}, tests: {len(fold_test)}")
        if agg:
            print(f"  MAE {agg['mae'][0]:.4f}±{agg['mae'][1]:.4f} | "
                  f"RMSE {agg['rmse'][0]:.4f}±{agg['rmse'][1]:.4f} | "
                  f"R2 {agg['r2'][0]:.4f}±{agg['r2'][1]:.4f}")

    # Save summary JSON for downstream use
    summary = {}
    for roi, d in data.items():
        if d["agg"]:
            summary[str(roi)] = {
                "n_folds_with_test": len(d["fold_test"]),
                "test_loss_mean": d["agg"]["test_loss"][0],
                "test_loss_std": d["agg"]["test_loss"][1],
                "mae_mean": d["agg"]["mae"][0],
                "mae_std": d["agg"]["mae"][1],
                "rmse_mean": d["agg"]["rmse"][0],
                "rmse_std": d["agg"]["rmse"][1],
                "r2_mean": d["agg"]["r2"][0],
                "r2_std": d["agg"]["r2"][1],
            }
    with open(FIG_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ===== Figure 1: Per-ROI training curves (one panel per metric, mean over folds) =====
    metrics = [
        ("val_loss", "Val Loss (SmoothL1)", True),
        ("val_mae", "Val MAE (mg/dL)", True),
        ("val_rmse", "Val RMSE (mg/dL)", True),
        ("val_r2", "Val R\u00b2", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (key, label, lower_better) in zip(axes, metrics):
        for roi, d in data.items():
            all_epochs = []
            all_vals = []
            for fold_id, epochs in d["fold_epochs"].items():
                xs = [e["epoch"] for e in epochs]
                ys = [e[key] for e in epochs]
                ax.plot(xs, ys, color=palette[roi], alpha=0.18, linewidth=0.9)
                all_epochs.append(xs)
                all_vals.append(ys)
            if all_vals:
                max_len = max(len(v) for v in all_vals)
                padded = np.full((len(all_vals), max_len), np.nan)
                for i, v in enumerate(all_vals):
                    padded[i, :len(v)] = v
                mean_curve = np.nanmean(padded, axis=0)
                std_curve = np.nanstd(padded, axis=0)
                xs = np.arange(1, max_len + 1)
                ax.plot(xs, mean_curve, color=palette[roi], linewidth=2.2,
                        label=f"ROI {roi}")
                ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                                color=palette[roi], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Per-ROI Validation Curves (mean ± std across folds)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "curves_by_roi.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR/'curves_by_roi.png'}")

    # ===== Figure 2: Per-fold test metrics (MAE / RMSE / R2) =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metric_keys = [("mae", "Test MAE (mg/dL)"), ("rmse", "Test RMSE (mg/dL)"), ("r2", "Test R\u00b2")]
    n_folds = 10
    x = np.arange(1, n_folds + 1)
    width = 0.2
    for ax, (key, label) in zip(axes, metric_keys):
        for i, roi in enumerate(ROI_SIZES):
            d = data.get(roi)
            if not d or not d["fold_test"]:
                continue
            ys = [d["fold_test"].get(f, {}).get(key, np.nan) for f in range(1, n_folds + 1)]
            ax.bar(x + (i - 1.5) * width, ys, width=width, color=palette[roi],
                   label=f"ROI {roi}", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in x])
        ax.set_xlabel("Fold")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-Fold Test Metrics by ROI Size", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_per_fold.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR/'test_per_fold.png'}")

    # ===== Figure 3: Cross-ROI comparison (mean ± std of test metrics) =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label) in zip(axes, metric_keys):
        means, stds, rois = [], [], []
        for roi in ROI_SIZES:
            d = data.get(roi)
            if d and d["agg"]:
                means.append(d["agg"][key][0])
                stds.append(d["agg"][key][1])
                rois.append(roi)
        if not rois:
            continue
        bars = ax.bar([str(r) for r in rois], means, yerr=stds, capsize=6,
                      color=[palette[r] for r in rois], alpha=0.85,
                      edgecolor="black", linewidth=0.7)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.4f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("ROI size")
        ax.set_ylabel(label)
        ax.set_title(f"{label} (mean ± std over 10 folds)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Cross-ROI Test Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_roi_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR/'cross_roi_comparison.png'}")

    # ===== Figure 4: Heatmap of test metrics (fold x ROI) =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (key, label) in zip(axes, metric_keys):
        mat = np.full((len(ROI_SIZES), n_folds), np.nan)
        for i, roi in enumerate(ROI_SIZES):
            d = data.get(roi)
            if not d:
                continue
            for f in range(1, n_folds + 1):
                if f in d["fold_test"] and key in d["fold_test"][f]:
                    mat[i, f - 1] = d["fold_test"][f][key]
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(n_folds))
        ax.set_xticklabels([f"F{i+1}" for i in range(n_folds)])
        ax.set_yticks(np.arange(len(ROI_SIZES)))
        ax.set_yticklabels([f"ROI {r}" for r in ROI_SIZES])
        ax.set_title(label)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < np.nanmean(mat) else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Test Metrics Heatmap (fold × ROI)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR/'metrics_heatmap.png'}")

    # ===== Figure 5: Train vs Val loss for best ROI =====
    best_roi = None
    if data:
        scored = [(r, d["agg"]["mae"][0]) for r, d in data.items() if d["agg"]]
        if scored:
            best_roi = min(scored, key=lambda x: x[1])[0]
    if best_roi is not None:
        d = data[best_roi]
        fig, ax = plt.subplots(figsize=(8, 5))
        for fold_id, epochs in d["fold_epochs"].items():
            xs = [e["epoch"] for e in epochs]
            train = [e["train_loss"] for e in epochs]
            val = [e["val_loss"] for e in epochs]
            ax.plot(xs, train, color=palette[best_roi], alpha=0.15, linewidth=0.8)
            ax.plot(xs, val, color="black", alpha=0.18, linewidth=0.8)
        # mean curves
        all_train, all_val = [], []
        for epochs in d["fold_epochs"].values():
            all_train.append([e["train_loss"] for e in epochs])
            all_val.append([e["val_loss"] for e in epochs])
        max_len = max(len(v) for v in all_train)
        pad_t = np.full((len(all_train), max_len), np.nan)
        pad_v = np.full((len(all_val), max_len), np.nan)
        for i, (t, v) in enumerate(zip(all_train, all_val)):
            pad_t[i, :len(t)] = t
            pad_v[i, :len(v)] = v
        xs = np.arange(1, max_len + 1)
        ax.plot(xs, np.nanmean(pad_t, axis=0), color=palette[best_roi],
                linewidth=2.4, label="Train Loss (mean)")
        ax.plot(xs, np.nanmean(pad_v, axis=0), color="black",
                linewidth=2.4, label="Val Loss (mean)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (SmoothL1)")
        ax.set_title(f"Train vs Val Loss — ROI {best_roi} (best by MAE)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"train_vs_val_roi{best_roi}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {FIG_DIR/f'train_vs_val_roi{best_roi}.png'}")

    print("\n[DONE] All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()