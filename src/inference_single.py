"""
Inference script: load best checkpoint của convnext_tiny_pretrain (Regression_wb_region)
và dự đoán nồng độ Bilirubin (blood mg/dL) từ 1 ảnh đầu vào.

Pipeline (khớp với lúc train trong src/utils/dataloader.py):
    1. Đọc ảnh gốc, convert sang RGB
    2. Center crop 224x224 ở giữa ảnh (zero-pad nếu ảnh nhỏ hơn 224)
    3. ToTensor + Normalize với ImageNet mean/std (model pretrained ImageNet)
    4. Forward qua ConvNeXtTinyRegression
    5. Trả về prediction (mg/dL)
    6. Save ROI (và tùy chọn: ảnh gốc có vẽ bbox ROI) để debug trực quan

Sử dụng:
    python src/inference_single.py \
        --image datasets/images/0001-1.jpg \
        --checkpoint src/checkpoint/Regression_wb_region/convnext_tiny_pretrain/checkpoints/fold9_best.pth \
        --roi_size 224 \
        --device cuda
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Đảm bảo import được src.models.* khi chạy từ root project
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models.convnext_tiny import ConvNeXtTinyRegression  # noqa: E402


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def center_crop_roi(image: Image.Image, roi_size: int) -> Image.Image:
    """Crop vùng trung tâm ảnh thành hình vuông roi_size x roi_size.
    Nếu ảnh nhỏ hơn roi_size thì zero-pad trước khi crop."""
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


def get_roi_bbox(image: Image.Image, roi_size: int) -> tuple:
    """Tính tọa độ bounding box của ROI center-crop trên ảnh gốc.
    Trả về (left, top, right, bottom). Nếu ảnh nhỏ hơn roi_size, bbox có thể âm.
    """
    w, h = image.size
    left = (w - roi_size) // 2
    top = (h - roi_size) // 2
    return (left, top, left + roi_size, top + roi_size)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load ConvNeXtTinyRegression từ checkpoint.
    Trả về (model, info_dict) với info_dict chứa metadata của checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Khởi tạo model với pretrained=False vì ta load state_dict đã train rồi.
    # (Tránh download weights không cần thiết và không có tác dụng vì sẽ bị overwrite.)
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

    # strict=True để báo lỗi nếu kiến trúc checkpoint không khớp ConvNeXtTinyRegression
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image = Image.open(image_path).convert("RGB")
    roi = center_crop_roi(image, roi_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(roi)  # (3, roi_size, roi_size)
    return tensor.unsqueeze(0)  # (1, 3, roi_size, roi_size)


def denormalize_tensor(tensor_chw: torch.Tensor) -> np.ndarray:
    """Denormalize tensor (3, H, W) đã normalize ImageNet -> numpy uint8 RGB (H, W, 3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor_chw.cpu() * std + mean
    img = img.clamp(0, 1)
    # (3, H, W) -> (H, W, 3)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def save_roi(roi_tensor_chw: torch.Tensor, save_path: str) -> str:
    """Lưu ROI (đã denormalize) ra file. Trả về đường dẫn tuyệt đối."""
    arr = denormalize_tensor(roi_tensor_chw)
    roi_img = Image.fromarray(arr, mode="RGB")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    roi_img.save(save_path)
    return os.path.abspath(save_path)


def save_original_with_bbox(
    image_path: str,
    roi_size: int,
    save_path: str,
    prediction: float = None,
    ground_truth: float = None,
    bbox_color: tuple = (255, 0, 0),
    bbox_width: int = 4,
) -> str:
    """Lưu ảnh gốc kèm bounding box ROI và label. Trả về đường dẫn tuyệt đối."""
    orig = Image.open(image_path).convert("RGB")
    bbox = get_roi_bbox(orig, roi_size)
    # Clip bbox vào biên ảnh (nếu ROI > ảnh thì bbox có thể âm/rộng hơn ảnh)
    left = max(0, bbox[0])
    top = max(0, bbox[1])
    right = min(orig.size[0], bbox[2])
    bottom = min(orig.size[1], bbox[3])

    draw = ImageDraw.Draw(orig)
    draw.rectangle([left, top, right, bottom], outline=bbox_color, width=bbox_width)

    # Label text trên góc trên-trái
    lines = [f"Pred: {prediction:.2f} mg/dL" if prediction is not None else None,
             f"GT  : {ground_truth:.2f} mg/dL" if ground_truth is not None else None,
             f"ROI : {roi_size}x{roi_size}"]
    lines = [l for l in lines if l is not None]
    text = "\n".join(lines)

    # Cố gắng dùng font mặc định, fallback nếu không có
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(14, roi_size // 14))
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Tính kích thước text box để vẽ nền đen
    try:
        text_bbox = draw.multiline_textbbox((left + 8, top + 8), text, font=font, spacing=2)
    except AttributeError:
        # Phiên bản PIL cũ không có multiline_textbbox
        text_bbox = draw.textbbox((left + 8, top + 8), text, font=font)
    bg_left = max(0, text_bbox[0] - 4)
    bg_top = max(0, text_bbox[1] - 4)
    bg_right = min(orig.size[0], text_bbox[2] + 4)
    bg_bottom = min(orig.size[1], text_bbox[3] + 4)
    draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=(0, 0, 0))
    draw.multiline_text((left + 8, top + 8), text, fill=(255, 255, 255), font=font, spacing=2)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    orig.save(save_path)
    return os.path.abspath(save_path)


@torch.no_grad()
def predict(
    image_path: str,
    checkpoint_path: str,
    roi_size: int = 224,
    device: str = "cpu",
    save_roi_path: str = None,
    save_bbox_path: str = None,
    ground_truth: float = None,
) -> dict:
    """Predict bilirubin cho 1 ảnh.

    Args:
        image_path: Đường dẫn tới ảnh đầu vào.
        checkpoint_path: Đường dẫn tới file .pth.
        roi_size: Kích thước ROI (default 224).
        device: 'cpu' hoặc 'cuda'.
        save_roi_path: Lưu ROI đã crop ra file này.
        save_bbox_path: Lưu ảnh gốc có vẽ bbox ROI + label ra file này.
        ground_truth: Giá trị bilirubin thực tế (mg/dL). Nếu truyền vào, sẽ tính thêm
            các metric sai số (Error, Absolute Error, Squared Error, Relative Error %).

    Returns:
        dict với keys: 'prediction_mg_dL', 'roi_size', 'image_size', 'ckpt_info',
        'saved_files', và nếu có ground_truth thì thêm 'error_metrics'.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model, info = load_model(checkpoint_path, device_obj)

    # Preprocess
    tensor = preprocess_image(image_path, roi_size=roi_size).to(device_obj)

    # Predict
    pred = model(tensor)  # (1,) regression output
    prediction = float(pred.squeeze(0).cpu().item())

    # Lấy kích thước ảnh gốc để log
    with Image.open(image_path) as orig_img:
        orig_w, orig_h = orig_img.size

    # Save ROI + bbox nếu được yêu cầu
    saved_files = {}
    if save_roi_path:
        saved_files["roi"] = save_roi(tensor[0], save_roi_path)
    if save_bbox_path:
        saved_files["bbox"] = save_original_with_bbox(
            image_path=image_path,
            roi_size=roi_size,
            save_path=save_bbox_path,
            prediction=prediction,
            ground_truth=ground_truth,
        )

    result = {
        "image_path": os.path.abspath(image_path),
        "image_size": (orig_w, orig_h),
        "roi_size": roi_size,
        "roi_bbox": get_roi_bbox(Image.new("RGB", (orig_w, orig_h)), roi_size),
        "prediction_mg_dL": prediction,
        "checkpoint": os.path.abspath(checkpoint_path),
        "ckpt_info": info,
        "device": str(device_obj),
        "ground_truth_mg_dL": ground_truth,
        "error_metrics": None,
        "saved_files": saved_files,
    }

    # Tính sai số nếu có ground truth
    if ground_truth is not None:
        result["error_metrics"] = compute_error_metrics(prediction, ground_truth)

    return result


def compute_error_metrics(prediction: float, ground_truth: float) -> dict:
    """Tính các metric sai số giữa prediction và ground truth.

    Returns:
        dict với keys: error, absolute_error, squared_error, relative_error_pct,
        rmse (1 sample), within_2mg (bool), within_20pct (bool).
    """
    error = prediction - ground_truth
    absolute_error = abs(error)
    squared_error = error ** 2
    relative_error_pct = (absolute_error / ground_truth * 100.0) if ground_truth != 0 else float("inf")
    rmse_sample = math.sqrt(squared_error)

    return {
        "error": error,
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "relative_error_pct": relative_error_pct,
        "rmse": rmse_sample,
        "within_2mg": absolute_error <= 2.0,
        "within_20pct": relative_error_pct <= 20.0,
    }


def lookup_ground_truth(image_path: str, csv_path: str, label_col: str = "blood(mg/dL)") -> dict:
    """Tra cứu ground truth (bilirubin) và thông tin fold/split của ảnh trong CSV.

    Trả về dict {'gt': float|None, 'matched_rows': list[dict]|None, 'matched': bool}.
    """
    if not os.path.exists(csv_path):
        return {"gt": None, "matched_rows": None, "matched": False}

    df = pd.read_csv(csv_path)
    fname = os.path.basename(image_path)

    matches = df[df["image_idx"] == fname]
    if matches.empty:
        return {"gt": None, "matched_rows": None, "matched": False}

    # Một ảnh có thể xuất hiện nhiều lần (1 lần/fold). Lấy tất cả để báo cho user biết.
    return {
        "gt": float(matches.iloc[0][label_col]),
        "matched_rows": matches.to_dict("records"),
        "matched": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bilirubin từ 1 ảnh.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn tới ảnh đầu vào (vd: datasets/images/0001-1.jpg)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="src/checkpoint/Regression_wb_region/convnext_tiny_pretrain/checkpoints/fold9_best.pth",
        help="Đường dẫn tới file .pth (mặc định: fold9_best.pth - checkpoint tốt nhất)",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=224,
        help="Kích thước ROI center crop (default: 224)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device để chạy inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Thư mục lưu ROI (mặc định: cùng thư mục với ảnh gốc + suffix '_roi'). "
             "Tên file: <image_stem>__roi<size>.jpg và <image_stem>__bbox<size>.jpg.",
    )
    parser.add_argument(
        "--save_roi",
        type=str,
        default=None,
        help="Đường dẫn cụ thể để lưu ROI (override --output_dir cho ROI).",
    )
    parser.add_argument(
        "--save_bbox",
        type=str,
        default=None,
        help="Đường dẫn cụ thể để lưu ảnh gốc có vẽ bbox ROI (override --output_dir cho bbox).",
    )
    parser.add_argument(
        "--no_save_roi",
        action="store_true",
        help="Tắt lưu ROI mặc định (mặc định: tự động lưu ROI vào output_dir).",
    )
    parser.add_argument(
        "--gt",
        type=float,
        default=None,
        help="Ground truth bilirubin (mg/dL) để tính sai số. Nếu không truyền, dùng --csv để auto-lookup.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="datasets/split_10fold_blood.csv",
        help="CSV chứa ground truth (dùng khi --gt không truyền). Default: split_10fold_blood.csv",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Tắt auto-lookup ground truth từ CSV (kể cả khi --gt không truyền).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Xác định ground truth: ưu tiên --gt, fallback tra trong --csv
    ground_truth = args.gt
    gt_meta = None
    if ground_truth is None and not args.no_csv and args.csv:
        gt_meta = lookup_ground_truth(args.image, args.csv)
        if gt_meta["matched"]:
            ground_truth = gt_meta["gt"]

    # Xác định đường dẫn lưu ROI mặc định
    save_roi_path = args.save_roi
    save_bbox_path = args.save_bbox
    if not args.no_save_roi:
        # Tính đường dẫn mặc định dựa trên ảnh gốc
        abs_image = os.path.abspath(args.image)
        image_dir = os.path.dirname(abs_image)
        image_stem = Path(abs_image).stem
        image_ext = ".jpg"
        # Nếu truyền --output_dir thì dùng thư mục đó, không thì dùng thư mục ảnh gốc
        out_dir = args.output_dir if args.output_dir else image_dir
        os.makedirs(out_dir, exist_ok=True)

        if save_roi_path is None:
            save_roi_path = os.path.join(
                out_dir, f"{image_stem}__roi{args.roi_size}{image_ext}"
            )
        if save_bbox_path is None:
            save_bbox_path = os.path.join(
                out_dir, f"{image_stem}__bbox{args.roi_size}{image_ext}"
            )

    result = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        roi_size=args.roi_size,
        device=args.device,
        save_roi_path=save_roi_path,
        save_bbox_path=save_bbox_path,
        ground_truth=ground_truth,
    )

    print("=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)
    print(f"Image         : {result['image_path']}")
    print(f"Original size : {result['image_size']}")
    print(f"ROI size      : {result['roi_size']} x {result['roi_size']}")
    bbox = result["roi_bbox"]
    print(f"ROI bbox      : ({bbox[0]}, {bbox[1]}) -> ({bbox[2]}, {bbox[3]})")
    print(f"Device        : {result['device']}")
    print(f"Checkpoint    : {result['checkpoint']}")
    info = result["ckpt_info"]
    if info:
        print(f"  fold        : {info.get('fold')}")
        print(f"  epoch       : {info.get('epoch')}")
        print(f"  val_loss    : {info.get('val_loss')}")
        print(f"  val_mae     : {info.get('val_mae')}")
        print(f"  val_r2      : {info.get('val_r2')}")
    print("-" * 60)
    print(f"PREDICTION    : {result['prediction_mg_dL']:.4f} mg/dL")
    print("=" * 60)

    # Phần tính sai số (nếu có ground truth)
    err = result["error_metrics"]
    if err is not None:
        gt = result["ground_truth_mg_dL"]
        sign = "+" if err["error"] >= 0 else ""
        direction = "OVER" if err["error"] > 0 else ("UNDER" if err["error"] < 0 else "EXACT")
        print()
        print("=" * 60)
        print("ERROR METRICS (vs ground truth)")
        print("=" * 60)
        print(f"Ground truth  : {gt:.4f} mg/dL")
        print(f"Prediction    : {result['prediction_mg_dL']:.4f} mg/dL")
        print(f"Direction     : {direction}  (model {'over' if err['error'] > 0 else 'under'}-estimates)")
        print("-" * 60)
        print(f"Error (signed): {sign}{err['error']:.4f} mg/dL")
        print(f"|Error| (MAE) : {err['absolute_error']:.4f} mg/dL")
        print(f"Squared Error : {err['squared_error']:.4f}")
        print(f"RMSE (1 samp) : {err['rmse']:.4f} mg/dL")
        print(f"Rel. Error    : {err['relative_error_pct']:.2f} %")
        print("-" * 60)
        print(f"Within ±2 mg/dL   : {'YES' if err['within_2mg'] else 'NO'}")
        print(f"Within ±20% (rel): {'YES' if err['within_20pct'] else 'NO'}")
        print("=" * 60)

        # Cảnh báo nếu ảnh nằm trong train set của fold trong checkpoint
        if gt_meta and gt_meta.get("matched_rows"):
            ckpt_fold = info.get("fold") if info else None
            train_rows = [r for r in gt_meta["matched_rows"] if r.get("fold") == ckpt_fold and r.get("split") == "train"]
            if train_rows and ckpt_fold is not None:
                print()
                print(f"[WARN] Ảnh này thuộc TRAIN set của fold {ckpt_fold} ->")
                print("       prediction có thể bị over-optimistic. Để đánh giá công bằng,")
                print("       hãy dùng checkpoint mà fold chứa ảnh trong val/test.")
                print("       (vd fold7_best.pth nếu ảnh nằm val của fold 7)")
    else:
        print()
        print("[INFO] Không có ground truth -> không tính sai số.")
        print("       Truyền --gt <value> hoặc dùng --csv <path> để auto-lookup.")

    # Phần thông báo file đã lưu
    saved = result["saved_files"]
    if saved:
        print()
        print("=" * 60)
        print("SAVED FILES (ROI / Visualization)")
        print("=" * 60)
        if "roi" in saved:
            print(f"ROI cropped   : {saved['roi']}")
        if "bbox" in saved:
            print(f"Image + bbox  : {saved['bbox']}")
        print("=" * 60)


if __name__ == "__main__":
    main()