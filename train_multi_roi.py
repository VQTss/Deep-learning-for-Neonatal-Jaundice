#!/usr/bin/env python3
"""
Script train ConvNeXtTinyRegression với nhiều mức ROI size khác nhau.
Mỗi experiment sẽ được lưu trong thư mục riêng để dễ so sánh.
"""

import os
import sys
import subprocess

# Cấu hình các mức ROI size cần train
# TODO: Điều chỉnh danh sách này theo nhu cầu của bạn
ROI_SIZES = [96, 128, 160, 224]

# Các tham số cố định
MODEL_NAME = "convnext_tiny"
PRETRAINED = True
EPOCHS = 100
BATCH_SIZE = 32
N_FOLDS = 10
SPLIT_MODE = "10fold"  # hoặc "holdout"

# Data path
DATA_PATH = "./datasets"
DATA_CSV = "split_10fold_blood.csv"
DATA_IMAGE = "datasets/images_wb_region"  # thư mục ảnh gốc

# Output base path
OUTPUT_BASE = "checkpoint/ROI_variants"


def train_with_roi_size(roi_size: int):
    """Train model với một ROI size cụ thể."""
    output_dir = f"{OUTPUT_BASE}/roi{roi_size}"

    cmd = [
        sys.executable,  # python interpreter hiện tại
        "src/train.py",
        "--model_name", MODEL_NAME,
        "--pretrained", str(PRETRAINED),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--n_folds", str(N_FOLDS),
        "--split_mode", SPLIT_MODE,
        "--data_path", DATA_PATH,
        "--data_csv", DATA_CSV,
        "--data_image", DATA_IMAGE,
        "--image_size", str(roi_size),
        "--output_root", output_dir,
        "--run_tag", f"roi{roi_size}",
    ]

    print("=" * 70)
    print(f"Starting training with ROI size = {roi_size}")
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    # Chạy training
    result = subprocess.run(cmd)

    return result.returncode


def main():
    print("=" * 70)
    print("ConvNeXtTinyRegression - Multi-ROI Size Training")
    print(f"ROI sizes: {ROI_SIZES}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("=" * 70)

    results = {}

    for i, roi_size in enumerate(ROI_SIZES):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(ROI_SIZES)}] Training with ROI size = {roi_size}")
        print(f"{'='*70}")

        returncode = train_with_roi_size(roi_size)

        if returncode == 0:
            results[roi_size] = "SUCCESS"
            print(f"✓ ROI size {roi_size} completed successfully")
        else:
            results[roi_size] = "FAILED"
            print(f"✗ ROI size {roi_size} failed with return code {returncode}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for roi_size, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {symbol} ROI {roi_size}: {status}")

    # Check results từ các fold
    print("\n" + "=" * 70)
    print("CHECKING RESULTS")
    print("=" * 70)
    for roi_size in ROI_SIZES:
        output_dir = f"{OUTPUT_BASE}/roi{roi_size}"
        if os.path.exists(output_dir):
            # Đọc kết quả từ fold results nếu có
            fold_results = []
            for fold in range(1, N_FOLDS + 1):
                result_file = os.path.join(output_dir, f"fold{fold:02d}_result.json")
                if os.path.exists(result_file):
                    import json
                    with open(result_file) as f:
                        fold_results.append(json.load(f))

            if fold_results:
                import numpy as np
                val_losses = [r["val_loss"] for r in fold_results]
                best_epochs = [r["best_epoch"] for r in fold_results]
                print(f"\nROI {roi_size}:")
                print(f"  Mean Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
                print(f"  Mean Best Epoch: {np.mean(best_epochs):.1f}")
                print(f"  Per-fold Val Losses: {[f'{v:.4f}' for v in val_losses]}")


if __name__ == "__main__":
    main()
