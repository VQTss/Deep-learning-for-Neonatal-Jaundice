import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


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


def get_train_transform(image_size: int = 224):
    """Transform cho tập train — augmentation NHẸ, tránh phá vỡ tín hiệu màu da."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_valid_transform(image_size: int = 224):
    """Transform cho val/test — KHÔNG augment."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Giữ lại tên cũ để tương thích ngược, nhưng trỏ về valid transform
def get_default_transform(image_size: int = 224):
    return get_valid_transform(image_size)


# ==================== Dataset ====================

class NeonatalJaundiceDataset(Dataset):
    """
    Dataset cho bài toán Regression dự đoán nồng độ Bilirubin (blood mg/dL)
    Tự động cắt ROI trung tâm ảnh (Center Crop)
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        roi_size: int = 224,
        split: str = None,
        fold: int = None,
        transform=None
    ):
        self.image_dir = image_dir
        self.roi_size = roi_size
        self.split = split

        # Tự động chọn augment transform nếu là train và transform=None
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transform(roi_size)
        else:
            self.transform = get_valid_transform(roi_size)

        df = pd.read_csv(csv_path)

        # Lọc theo fold
        if fold is not None:
            df = df[df["fold"] == fold]

        # Lọc theo split (train/val/test) - chỉ khi fold cũng được chỉ định
        if split is not None and fold is not None:
            df = df[df["split"] == split]

        self.df = df.reset_index(drop=True)
        print(f"[Dataset] Loaded {len(self.df)} samples | roi_size={roi_size} | split={split} | fold={fold}")

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
            image = self.transform(image)

        return image, label


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
    pin_memory: bool = True
) -> DataLoader:
    """
    Tạo DataLoader với Auto Center ROI Crop.
    """
    dataset = NeonatalJaundiceDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        roi_size=roi_size,
        split=split,
        fold=fold,
        transform=None  # Để Dataset tự chọn transform theo split
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
    num_workers: int = 4
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
        pin_memory=False  # Tắt pin_memory để tránh CUDA crash giữa các fold
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
        pin_memory=False  # Tắt pin_memory để tránh CUDA crash giữa các fold
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
