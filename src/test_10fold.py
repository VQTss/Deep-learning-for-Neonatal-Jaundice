"""
Test 10-fold cross-validation bằng cách chạy inference trên test set của từng fold.

Cho mỗi fold f trong 1..10:
    1. Load checkpoint fold{f}_best.pth
    2. Lấy test set (split == 'test') của fold f từ CSV
    3. Predict trên từng ảnh (ảnh từ images_wb_region/)
    4. Tính metrics: MAE, RMSE, R^2, within ±2mg, within ±20%
    5. Lưu predictions ra CSV

Tổng hợp: in bảng metrics của 10 fold + trung bình ± std.

Sử dụng:
    python src/test_10fold.py \
        --checkpoint_dir src/checkpoint/Regression_wb_region/convnext_tiny_pretrain/checkpoints \
        --images_dir datasets/images_wb_region \
        --csv datasets/split_10fold_blood.csv \
        --device cuda \
        --output_dir src/checkpoint/Regression_wb_region/convnext_tiny_pretrain/test_10fold
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# Đảm bảo import được src.models.* khi chạy từ root project
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models.convnext_tiny import ConvNeXtTinyRegression  # noqa: E402


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def center_crop_roi(image: Image.Image, roi_size: int) -> Image.Image:
    """Center crop thành roi_size x roi_size (zero-pad nếu ảnh nhỏ hơn)."""
    w, h = image.size
    pad_left = max(0, (roi_size - w) // 2)
    pad_top = max(0, (roi_size - h) // 2)
    pad_right = max(0, roi_size - w - pad_left)
    pad_bottom = max(0, roi_size - h - pad_top)

    if pad_left or pad_top or pad_right or pad_bottom:
        new_img = Image.new(
            "RGB",
            (w + pad_left + pad_right, h + pad_top + pad_bottom),
            (0, 0, 0),
        )
        new_img.paste(image, (pad_left, pad_top))
        image = new_img
        w, h = image.size

    left = (w - roi_size) // 2
    top = (h - roi_size) // 2
    return image.crop((left, top, left + roi_size, top + roi_size))


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load ConvNeXtTinyRegression từ checkpoint. Trả về (model, info_dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConvNeXtTinyRegression(pretrained=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        info = {
            "fold": ckpt.get("fold"),
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "val_mae": ckpt.get("val_mae"),
            "val_r2": ckpt.get("val_r2"),
        }
    else:
        state_dict = ckpt
        info = {}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys khi load checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys khi load checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model, info


def preprocess_image(image_path: str, roi_size: int = 224) -> torch.Tensor:
    """Đọc ảnh, center crop roi_size x roi_size, ToTensor + Normalize ImageNet."""
    image = Image.open(image_path).convert("RGB")
    roi = center_crop_roi(image, roi_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(roi)
    return tensor.unsqueeze(0)


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    """Tính các metric regression chuẩn."""
    err = preds - gts
    abs_err = np.abs(err)
    sq_err = err ** 2

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    r2 = float(1.0 - sq_err.sum() / np.sum((gts - gts.mean()) ** 2))
    bias = float(err.mean())  # mean error (signed): > 0 nghĩa là model over-predict
    median_ae = float(np.median(abs_err))
    max_ae = float(abs_err.max())
    within_2mg = float((abs_err <= 2.0).mean()) * 100.0  # %
    within_3mg = float((abs_err <= 3.0).mean()) * 100.0
    within_20pct = float((abs_err / np.maximum(gts, 1e-6) <= 0.20).mean()) * 100.0

    return {
        "n": int(len(preds)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "median_ae": median_ae,
        "max_ae": max_ae,
        "within_2mg_pct": within_2mg,
        "within_3mg_pct": within_3mg,
        "within_20pct_pct": within_20pct,
    }


@torch.no_grad()
def predict_batch(model, image_paths: list, roi_size: int, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Predict hàng loạt theo batch để tăng tốc. Trả về numpy array (N,)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    preds = np.zeros(len(image_paths), dtype=np.float32)
    n = len(image_paths)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for p in image_paths[start:end]:
            img = Image.open(p).convert("RGB")
            roi = center_crop_roi(img, roi_size)
            batch_imgs.append(transform(roi))
        batch = torch.stack(batch_imgs, dim=0).to(device)
        out = model(batch)
        # Output có thể là (B, 1) hoặc (B,) tuỳ model impl -> flatten về 1D
        out = out.reshape(-1).cpu().numpy()
        preds[start:end] = out

    return preds


def test_one_fold(
    fold: int,
    checkpoint_dir: str,
    images_dir: str,
    df: pd.DataFrame,
    device: torch.device,
    roi_size: int = 224,
    batch_size: int = 32,
    output_dir: str = None,
    save_predictions: bool = True,
) -> dict:
    """Test trên 1 fold, trả về dict metrics."""
    ckpt_path = os.path.join(checkpoint_dir, f"fold{fold}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    model, ckpt_info = load_model(ckpt_path, device)
    print(f"  [fold{fold}] Loaded {ckpt_path}  epoch={ckpt_info.get('epoch')}  "
          f"val_mae={ckpt_info.get('val_mae'):.4f}" if ckpt_info.get("val_mae") is not None
          else f"  [fold{fold}] Loaded {ckpt_path}")

    # Lấy test set của fold này (split == 'test' VÀ fold == fold)
    test_df = df[(df["fold"] == fold) & (df["split"] == "test")].copy()
    if test_df.empty:
        raise ValueError(f"Không có test rows nào cho fold {fold}")

    # Build full paths
    test_df["image_path"] = test_df["image_idx"].apply(lambda x: os.path.join(images_dir, x))

    # Kiểm tra file tồn tại
    missing = test_df[~test_df["image_path"].apply(os.path.exists)]
    if not missing.empty:
        raise FileNotFoundError(f"  [fold{fold}] Thiếu {len(missing)} ảnh trên disk (vd: {missing.iloc[0]['image_idx']})")

    image_paths = test_df["image_path"].tolist()
    gts = test_df["blood(mg/dL)"].astype(float).values

    # Predict theo batch
    t0 = time.time()
    preds = predict_batch(model, image_paths, roi_size, device, batch_size=batch_size)
    elapsed = time.time() - t0

    metrics = compute_metrics(preds, gts)
    metrics["fold"] = fold
    metrics["ckpt_epoch"] = ckpt_info.get("epoch")
    metrics["ckpt_val_mae"] = ckpt_info.get("val_mae")
    metrics["ckpt_val_r2"] = ckpt_info.get("val_r2")
    metrics["elapsed_sec"] = elapsed
    # Cast numpy float32 -> python float để JSON-safe
    metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in metrics.items()}
    metrics["fold"] = int(metrics["fold"])
    if metrics["ckpt_epoch"] is not None:
        metrics["ckpt_epoch"] = int(metrics["ckpt_epoch"])
    if metrics["ckpt_val_mae"] is not None:
        metrics["ckpt_val_mae"] = float(metrics["ckpt_val_mae"])
    if metrics["ckpt_val_r2"] is not None:
        metrics["ckpt_val_r2"] = float(metrics["ckpt_val_r2"])

    print(f"  [fold{fold}] n={metrics['n']}  "
          f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  "
          f"bias={metrics['bias']:+.4f}  within2mg={metrics['within_2mg_pct']:.1f}%  "
          f"within20%={metrics['within_20pct_pct']:.1f}%  ({elapsed:.1f}s)")

    # Lưu predictions per-fold
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, f"fold{fold}_predictions.csv")
        save_df = test_df[["patient_id", "image_idx", "blood(mg/dL)", "fold", "split"]].copy()
        save_df["prediction_mg_dL"] = preds
        save_df["error_mg_dL"] = preds - gts
        save_df["abs_error_mg_dL"] = np.abs(preds - gts)
        save_df["rel_error_pct"] = np.abs(preds - gts) / np.maximum(gts, 1e-6) * 100.0
        save_df.to_csv(out_csv, index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test 10-fold cross-validation.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="src/checkpoint/Regression_wb_region/convnext_tiny_pretrain/checkpoints",
        help="Thư mục chứa fold{1..10}_best.pth",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="datasets/images_wb_region",
        help="Thư mục ảnh đã white-balance (region)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="datasets/split_10fold_blood.csv",
        help="CSV chứa fold/split/ground-truth",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=224,
        help="Kích thước ROI center crop",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size khi predict",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1-10",
        help="Các fold cần test, vd '1-10' hoặc '1,2,5' hoặc '7'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Thư mục lưu predictions CSV và summary JSON",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Tắt lưu file CSV predictions per-fold",
    )
    return parser.parse_args()


def parse_folds(spec: str) -> list:
    """Parse '1-10' hoặc '1,2,5' hoặc '7' thành list[int]."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    args = parse_args()
    folds = parse_folds(args.folds)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Folds  : {folds}")
    print(f"CKPT   : {args.checkpoint_dir}")
    print(f"Images : {args.images_dir}")
    print(f"CSV    : {args.csv}")
    print(f"Output : {args.output_dir or '(none)'}")

    df = pd.read_csv(args.csv)
    print(f"Loaded CSV: {len(df)} rows, "
          f"{df['image_idx'].nunique()} unique images, "
          f"folds {sorted(df['fold'].unique())}")

    all_metrics = []
    t_start = time.time()
    for fold in folds:
        print(f"\n--- fold{fold} ---")
        m = test_one_fold(
            fold=fold,
            checkpoint_dir=args.checkpoint_dir,
            images_dir=args.images_dir,
            df=df,
            device=device,
            roi_size=args.roi_size,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_predictions=not args.no_save_predictions,
        )
        all_metrics.append(m)
    total_time = time.time() - t_start

    # In bảng tổng hợp
    print("\n" + "=" * 90)
    print("10-FOLD CROSS-VALIDATION SUMMARY  (test set)")
    print("=" * 90)
    header = f"{'fold':<6}{'n':<6}{'MAE':<10}{'RMSE':<10}{'R^2':<10}{'bias':<10}{'within2mg':<12}{'within20%':<12}"
    print(header)
    print("-" * 90)
    for m in all_metrics:
        print(f"{m['fold']:<6}{m['n']:<6}"
              f"{m['mae']:<10.4f}{m['rmse']:<10.4f}{m['r2']:<10.4f}"
              f"{m['bias']:<+10.4f}{m['within_2mg_pct']:<12.2f}{m['within_20pct_pct']:<12.2f}")

    # Mean ± std (across folds)
    arr = np.array([[m["mae"], m["rmse"], m["r2"], m["bias"], m["within_2mg_pct"], m["within_20pct_pct"]]
                    for m in all_metrics])
    print("-" * 90)
    mean = [float(x) for x in arr.mean(axis=0)]
    std = [float(x) for x in arr.std(axis=0)]
    print(f"{'mean':<6}{'-':<6}"
          f"{mean[0]:<10.4f}{mean[1]:<10.4f}{mean[2]:<10.4f}"
          f"{mean[3]:<+10.4f}{mean[4]:<12.2f}{mean[5]:<12.2f}")
    print(f"{'std':<6}{'-':<6}"
          f"{std[0]:<10.4f}{std[1]:<10.4f}{std[2]:<10.4f}"
          f"{std[3]:<+10.4f}{std[4]:<12.2f}{std[5]:<12.2f}")
    print("=" * 90)
    print(f"Total time: {total_time:.1f}s")

    # Lưu summary JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        summary = {
            "config": {
                "checkpoint_dir": args.checkpoint_dir,
                "images_dir": args.images_dir,
                "csv": args.csv,
                "roi_size": args.roi_size,
                "folds": folds,
            },
            "per_fold": all_metrics,
            "mean": {
                "mae": mean[0],
                "rmse": mean[1],
                "r2": mean[2],
                "bias": mean[3],
                "within_2mg_pct": mean[4],
                "within_20pct_pct": mean[5],
            },
            "std": {
                "mae": std[0],
                "rmse": std[1],
                "r2": std[2],
                "bias": std[3],
                "within_2mg_pct": std[4],
                "within_20pct_pct": std[5],
            },
            "total_time_sec": float(total_time),
        }
        summary_path = os.path.join(args.output_dir, "summary_10fold.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()