import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ==================== Color Space Helpers ====================

VALID_COLOR_SPACES = ("rgb", "hsv", "lab", "ycbcr")

# Mean/std cho từng color space, dùng cho torchvision.transforms.Normalize.
# Lưu ý: torchvision Normalize nhận mean/std theo thứ tự channel (CHW format)
# và hoạt động trên tensor đã chia 255 (range [0,1]).
# Nên ở đây mean/std là giá trị ở range [0,1].
COLOR_SPACE_STATS: dict = {
    # ImageNet RGB stats (chuẩn cho pretrained ImageNet)
    "rgb":    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # HSV (OpenCV range: H 0-179, S/V 0-255; chia 255 về [0,1])
    "hsv":    ([0.500, 0.500, 0.500], [0.250, 0.250, 0.250]),
    # LAB (OpenCV L 0-255, a/b 0-255; đã được scale về [0,1])
    "lab":    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # dùng ImageNet-style, placeholder
    # YCbCr (Y 16-235, Cb/Cr 16-240; chia 255 về [0,1])
    "ycbcr":  ([0.500, 0.500, 0.500], [0.250, 0.250, 0.250]),
}


def convert_color_space(rgb_array: np.ndarray, color_space: str) -> np.ndarray:
    """
    Convert RGB uint8 ndarray (H, W, 3) sang color space khác.
    Trả về uint8 ndarray (H, W, 3).
    """
    if color_space == "rgb":
        return rgb_array
    if color_space == "hsv":
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    if color_space == "lab":
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    if color_space == "ycbcr":
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2YCrCb)
    raise ValueError(f"Unknown color_space '{color_space}'. Valid: {VALID_COLOR_SPACES}")


# ==================== Helper Functions ====================

def center_crop_roi(image: Image.Image, roi_size: int) -> Image.Image:
    """Cắt vùng trung tâm ảnh thành hình vuông. Zero-pad nếu ảnh nhỏ hơn roi_size."""
    w, h = image.size
    rs = roi_size

    # Zero-pad nếu ảnh nhỏ hơn roi_size
    pad_left = max(0, (rs - w) // 2)
    pad_top = max(0, (rs - h) // 2)
    pad_right = max(0, rs - w - pad_left)
    pad_bottom = max(0, rs - h - pad_top)

    if pad_left or pad_top or pad_right or pad_bottom:
        new_img = Image.new(
            "RGB",
            (w + pad_left + pad_right, h + pad_top + pad_bottom),
            (0, 0, 0),
        )
        new_img.paste(image, (pad_left, pad_top))
        image = new_img
        w, h = image.size

    left = (w - rs) // 2
    top = (h - rs) // 2
    right = left + rs
    bottom = top + rs
    return image.crop((left, top, right, bottom))


def get_train_transform(image_size: int = 224, color_space: str = "rgb"):
    """Transform cho tập train — augmentation NHẸ, tránh phá vỡ tín hiệu màu da.

    Args:
        image_size: Kích thước ảnh (không dùng trực tiếp ở đây, giữ để tương thích).
        color_space: Tên color space (dùng để lấy mean/std cho Normalize).

    ColorJitter bị tắt khi color_space == "rgb" (bài toán không color_space):
    - Giữ nguyên tính chất màu da, không làm shift màu/tương phản.
    ColorJitter chỉ bị tắt khi dùng non-RGB spaces vì không áp dụng được.
    """
    mean, std = COLOR_SPACE_STATS[color_space]
    if color_space == "rgb":
        # Không ColorJitter: giữ nguyên tín hiệu màu gốc
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_valid_transform(image_size: int = 224, color_space: str = "rgb"):
    """Transform cho val/test — KHÔNG augment."""
    mean, std = COLOR_SPACE_STATS[color_space]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_train_transform_multibranch(image_size: int = 224, spaces: list = None):
    """Transform cho multi-branch mode (2 color spaces, 6 channels input).
    Skip ColorJitter vì torchvision ColorJitter chỉ hỗ trợ 3-channel input.
    """
    if spaces is None or len(spaces) != 2:
        raise ValueError(f"spaces must be list of 2 color spaces, got: {spaces}")
    mean = COLOR_SPACE_STATS[spaces[0]][0] + COLOR_SPACE_STATS[spaces[1]][0]
    std = COLOR_SPACE_STATS[spaces[0]][1] + COLOR_SPACE_STATS[spaces[1]][1]
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_valid_transform_multibranch(image_size: int = 224, spaces: list = None):
    """Valid transform cho multi-branch mode."""
    if spaces is None or len(spaces) != 2:
        raise ValueError(f"spaces must be list of 2 color spaces, got: {spaces}")
    mean = COLOR_SPACE_STATS[spaces[0]][0] + COLOR_SPACE_STATS[spaces[1]][0]
    std = COLOR_SPACE_STATS[spaces[0]][1] + COLOR_SPACE_STATS[spaces[1]][1]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# Giữ lại tên cũ để tương thích ngược, nhưng trỏ về valid transform
def get_default_transform(image_size: int = 224):
    return get_valid_transform(image_size)


# ==================== Dataset ====================

class NeonatalJaundiceDataset(Dataset):
    """
    Dataset cho bài toán Regression dự đoán nồng độ Bilirubin (blood mg/dL)
    Tự động cắt ROI trung tâm ảnh (Center Crop)

    Hỗ trợ 2 mode:
    - Single: 1 color space, 3-channel output (RGB/HSV/LAB/YCbCr)
    - Multibranch: 2 color spaces, 6-channel output (concat theo channel dim)
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        roi_size: int = 224,
        split: str = None,
        fold: int = None,
        transform=None,
        color_space: str = "rgb",
        spaces: list = None,
    ):
        self.image_dir = image_dir
        self.roi_size = roi_size
        self.split = split
        self.color_space = color_space
        self.spaces = spaces  # None cho single mode, list[2] cho multibranch

        # Tự động chọn augment transform nếu là train và transform=None
        if transform is not None:
            self.transform = transform
        elif split == "train":
            if spaces is not None:
                self.transform = get_train_transform_multibranch(roi_size, spaces)
            else:
                self.transform = get_train_transform(roi_size, color_space)
        else:
            if spaces is not None:
                self.transform = get_valid_transform_multibranch(roi_size, spaces)
            else:
                self.transform = get_valid_transform(roi_size, color_space)

        df = pd.read_csv(csv_path)

        # Lọc theo fold
        if fold is not None:
            df = df[df["fold"] == fold]

        # Lọc theo split (train/val/test) - chỉ khi fold cũng được chỉ định
        if split is not None and fold is not None:
            df = df[df["split"] == split]

        self.df = df.reset_index(drop=True)
        mode = "multibranch" if spaces is not None else f"single:{color_space}"
        print(f"[Dataset] Loaded {len(self.df)} samples | roi_size={roi_size} | split={split} | fold={fold} | mode={mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_idx"])

        # Load ảnh gốc
        image = Image.open(img_path).convert("RGB")

        # === Tự động cắt ROI trung tâm ===
        image = center_crop_roi(image, self.roi_size)

        # Label (blood mg/dL)
        label = torch.tensor(row["blood(mg/dL)"], dtype=torch.float32)

        if self.transform:
            if self.spaces is not None:
                # Multi-branch mode: stack 2 color spaces thành 6 channels
                img_arr = np.array(image)  # (H, W, 3) RGB
                arr_s1 = convert_color_space(img_arr, self.spaces[0])
                arr_s2 = convert_color_space(img_arr, self.spaces[1])
                stacked = np.concatenate([arr_s1, arr_s2], axis=2)  # (H, W, 6)
                # Convert sang tensor uint8 (C, H, W) để apply transform đa-channel
                tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0
                # Apply HFlip/Rotation bằng torchvision transforms trên tensor
                # Vì self.transform đã được Compose với ToTensor + Normalize,
                # ta chỉ apply RandomHorizontalFlip + RandomRotation (augment-only).
                # Tuy nhiên, để code đơn giản và tận dụng pipeline, ta convert
                # tensor -> PIL 'RGB' fake (chỉ 3 channels) để chạy phần augment,
                # rồi lấy 6 channels lại. NHƯNG cách này phức tạp.
                # Cách đơn giản hơn: Apply augment bằng tay trên tensor 6-channel.
                if self.split == "train":
                    tensor = self._apply_basic_augment(tensor)
                # Normalize (đã được chia 255 trước đó)
                mean = torch.tensor(
                    COLOR_SPACE_STATS[self.spaces[0]][0] + COLOR_SPACE_STATS[self.spaces[1]][0]
                ).view(6, 1, 1)
                std = torch.tensor(
                    COLOR_SPACE_STATS[self.spaces[0]][1] + COLOR_SPACE_STATS[self.spaces[1]][1]
                ).view(6, 1, 1)
                tensor = (tensor - mean) / std
                image = tensor
            else:
                # Single mode: convert color space rồi wrap PIL để áp transform
                img_arr = np.array(image)  # (H, W, 3) RGB
                converted = convert_color_space(img_arr, self.color_space)
                image = Image.fromarray(converted)
                image = self.transform(image)

        return image, label

    def _apply_basic_augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply RandomHorizontalFlip + RandomRotation trên tensor đa-channel (C, H, W) float32 [0,1].
        ColorJitter đã được skip ở multi-branch mode."""
        # Horizontal flip (random p=0.5)
        if torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        # Random rotation (degrees=10) — dùng affine trên tensor
        # torchvision.transforms.RandomRotation có thể áp dụng trên tensor
        rot_transform = transforms.RandomRotation(degrees=10)
        tensor = rot_transform(tensor)
        return tensor


# ==================== DataLoader Factory ====================

def get_dataloader(
    csv_path: str,
    image_dir: str,
    roi_size: int = 224,
    split: str = None,
    fold: int = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    color_space: str = "rgb",
    spaces: list = None,
) -> DataLoader:
    """
    Tạo DataLoader với Auto Center ROI Crop.

    Args:
        color_space: Color space cho single-branch mode (rgb/hsv/lab/ycbcr).
        spaces: List 2 color spaces cho multibranch mode. Nếu != None, color_space bị bỏ qua.
    """
    dataset = NeonatalJaundiceDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        roi_size=roi_size,
        split=split,
        fold=fold,
        transform=None,  # Để Dataset tự chọn transform theo split
        color_space=color_space,
        spaces=spaces,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train")
    )


def get_10fold_dataloaders(
    csv_path: str,
    image_dir: str,
    fold: int,
    roi_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    color_space: str = "rgb",
    spaces: list = None,
):
    """
    Trả về train_loader và val_loader cho một fold cụ thể.
    Leave-Patient-Out CV:
    - Train: fold == current_fold VÀ split == "train"
    - Val: fold == current_fold VÀ split == "val"
    """
    train_loader = get_dataloader(
        csv_path=csv_path,
        image_dir=image_dir,
        roi_size=roi_size,
        split="train",
        fold=fold,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Tắt pin_memory để tránh CUDA crash giữa các fold
        color_space=color_space,
        spaces=spaces,
    )

    val_loader = get_dataloader(
        csv_path=csv_path,
        image_dir=image_dir,
        roi_size=roi_size,
        split="val",
        fold=fold,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Tắt pin_memory để tránh CUDA crash giữa các fold
        color_space=color_space,
        spaces=spaces,
    )

    return train_loader, val_loader


def get_fold_dataset_info(csv_path: str, fold: int, splits=("train", "val", "test")):
    """
    Trả về dict thông tin dataset của 1 fold: số patient, số images per split.
    """
    df = pd.read_csv(csv_path)
    fold_df = df[df["fold"] == fold].copy()

    info = {"fold": fold}
    for split in splits:
        s = fold_df[fold_df["split"] == split]
        info[f"{split}_patients"] = s["patient_id"].nunique()
        info[f"{split}_images"] = len(s)
        if "blood(mg/dL)" in s.columns:
            labels = s["blood(mg/dL)"].dropna()
            info[f"{split}_label_mean"] = round(labels.mean(), 2)
            info[f"{split}_label_std"] = round(labels.std(), 2)

    return info


def save_roi_for_fold(
    csv_path: str,
    image_dir: str,
    fold: int,
    roi_size: int,
    output_root: str,
    splits=("train", "val", "test"),
):
    """
    Lưu ROI (center crop) của từng fold ra disk để kiểm tra trực quan.
    Cấu trúc:
        output_root/
            foldXX/
                train/  *.png
                val/    *.png
                test/   *.png
                foldXX_roi_manifest.csv
        roi_manifest.csv  (gộp tất cả fold)
    """
    df = pd.read_csv(csv_path)
    fold_df = df[df["fold"] == fold].copy()
    fold_info = get_fold_dataset_info(csv_path, fold, splits=splits)
    if fold_df.empty:
        return fold_df, fold_info

    fold_dir = os.path.join(output_root, f"fold{fold:02d}")
    saved_rows = []

    for split in splits:
        split_df = fold_df[fold_df["split"] == split]
        if split_df.empty:
            continue

        split_dir = os.path.join(fold_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for _, row in tqdm(split_df.iterrows(), desc=f"[ROI] fold{fold:02d}/{split}", leave=False):
            img_path = os.path.join(image_dir, row["image_idx"])
            dst_path = os.path.join(split_dir, os.path.basename(row["image_idx"]))

            # Skip ROI extraction nếu file đã tồn tại (tránh lặp lại I/O khi train nhiều model)
            if os.path.exists(dst_path):
                saved_rows.append({
                    "fold": fold,
                    "split": split,
                    "image_idx": row["image_idx"],
                    "roi_path": dst_path,
                    "label": row.get("blood(mg/dL)", None),
                })
                continue

            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            roi = center_crop_roi(image, roi_size)
            roi.save(dst_path)

            saved_rows.append({
                "fold": fold,
                "split": split,
                "image_idx": row["image_idx"],
                "roi_path": dst_path,
                "label": row.get("blood(mg/dL)", None),
            })

    manifest = pd.DataFrame(saved_rows)
    manifest_path = os.path.join(fold_dir, f"fold{fold:02d}_roi_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    # Cập nhật manifest gộp
    all_manifest_path = os.path.join(output_root, "roi_manifest.csv")
    if os.path.exists(all_manifest_path):
        prev = pd.read_csv(all_manifest_path)
        prev = prev[prev["fold"] != fold]
        manifest = pd.concat([prev, manifest], ignore_index=True)
    manifest.to_csv(all_manifest_path, index=False)

    return fold_df, fold_info
