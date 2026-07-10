import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ==================== FFT Low-pass Denoise ====================

def _gaussian_lowpass_mask(size: int, d0: float) -> np.ndarray:
    """Tạo Gaussian low-pass mask kích thước size×size, cutoff d0 (pixels)."""
    cy = cx = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.exp(-(r ** 2) / (2.0 * d0 ** 2))


def fft_lowpass_denoise_pil(image: Image.Image, d0: float = 30.0) -> Image.Image:
    """Apply FFT Gaussian low-pass denoise lên PIL Image (RGB). Trả về PIL Image uint8."""
    arr = np.array(image).astype(np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    size = arr.shape[0]
    H = _gaussian_lowpass_mask(size, d0)
    clean = np.zeros_like(arr)
    for c in range(arr.shape[-1]):
        F = np.fft.fftshift(np.fft.fft2(arr[..., c]))
        clean[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8))


# ==================== Helper Functions ====================

def center_crop_roi(image: Image.Image, roi_size: int) -> Image.Image:
    """Cắt vùng trung tâm ảnh vuông roi_size×roi_size.

    Yêu cầu: ảnh đầu vào phải đã vuông và kích thước >= roi_size.
    Nếu ảnh nhỏ hơn roi_size sẽ raise ValueError (tránh zero-pad gây viền đen).
    """
    w, h = image.size
    if w != h:
        raise ValueError(f"center_crop_roi: ảnh phải vuông, nhận {w}x{h}")
    if w < roi_size:
        raise ValueError(f"center_crop_roi: ảnh {w}x{w} < roi_size={roi_size}")

    left = (w - roi_size) // 2
    top = (h - roi_size) // 2
    right = left + roi_size
    bottom = top + roi_size
    return image.crop((left, top, right, bottom))


def get_train_transform(image_size: int = 128):
    """Transform cho tập train — augmentation NHẸ, tránh phá vỡ tín hiệu màu da."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_valid_transform(image_size: int = 128):
    """Transform cho val/test — KHÔNG augment."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Giữ lại tên cũ để tương thích ngược, nhưng trỏ về valid transform
def get_default_transform(image_size: int = 128):
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
        roi_size: int = 128,
        split: str = None,
        fold: int = None,
        transform=None,
        use_fft: bool = False,
        fft_d0: float = 30.0,
        fft_cache_dir: str = None,
    ):
        self.image_dir = image_dir
        self.roi_size = roi_size
        self.split = split
        self.use_fft = use_fft
        self.fft_d0 = fft_d0
        self.fft_cache_dir = fft_cache_dir

        # Tự động chọn augment transform nếu là train và transform=None
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transform(roi_size)
        else:
            self.transform = get_valid_transform(roi_size)

        df = pd.read_csv(csv_path)

        # Lọc theo fold (fold=0 -> test_fixed trong holdout split)
        if fold is not None:
            df = df[df["fold"] == fold]

        # Lọc theo split (train/val/test/test_fixed)
        # Tách riêng khỏi filter fold để hỗ trợ split="test_fixed" với fold=0
        if split is not None:
            if "split" not in df.columns:
                raise ValueError(f"CSV '{csv_path}' missing 'split' column")
            df = df[df["split"] == split]

        self.df = df.reset_index(drop=True)
        fft_tag = f" | use_fft={use_fft} d0={fft_d0}" if use_fft else ""
        print(f"[Dataset] Loaded {len(self.df)} samples | roi_size={roi_size} | split={split} | fold={fold}{fft_tag}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_idx"])

        # Nếu cache FFT tồn tại, đọc thẳng từ cache (nhanh)
        if self.use_fft and self.fft_cache_dir:
            cached_path = os.path.join(self.fft_cache_dir, row["image_idx"])
            if os.path.exists(cached_path):
                image = Image.open(cached_path).convert("RGB")
            else:
                # Fallback: compute on-the-fly
                image = Image.open(img_path).convert("RGB")
                image = center_crop_roi(image, self.roi_size)
                image = fft_lowpass_denoise_pil(image, d0=self.fft_d0)
        else:
            # Load ảnh gốc và crop ROI
            image = Image.open(img_path).convert("RGB")
            image = center_crop_roi(image, self.roi_size)
            if self.use_fft:
                image = fft_lowpass_denoise_pil(image, d0=self.fft_d0)

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
    pin_memory: bool = True,
    use_fft: bool = False,
    fft_d0: float = 30.0,
    fft_cache_dir: str = None,
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
        transform=None,  # Để Dataset tự chọn transform theo split
        use_fft=use_fft,
        fft_d0=fft_d0,
        fft_cache_dir=fft_cache_dir,
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
    use_fft: bool = False,
    fft_d0: float = 30.0,
    fft_cache_dir: str = None,
):
    """
    Trả về train_loader và val_loader cho một fold cụ thể.

    2 chế độ (tự động detect):
    - LONG-FORMAT (split_10fold_blood.csv):
        Train: fold == current_fold VÀ split == "train"
        Val:   fold == current_fold VÀ split == "val"
        (Mỗi patient xuất hiện 10 lần, một cho mỗi fold)
    - WIDE-FORMAT (split_holdout.csv):
        Train: fold ∈ {1..10} AND fold != current_fold (lọc theo fold, không lọc split)
        Val:   fold == current_fold (không lọc split)
        (Mỗi patient xuất hiện 1 lần, split="train" placeholder cho pool)

    Auto-detect: nếu CSV có cột "split" với giá trị {"train","val","test"} cho fold
    hiện tại thì dùng long-format; ngược lại dùng wide-format.
    """
    df_check = pd.read_csv(csv_path)
    fold_df_check = df_check[df_check["fold"] == fold]
    has_long_format = (
        "split" in fold_df_check.columns
        and (fold_df_check["split"] == "val").any()
    )

    if has_long_format:
        # Long-format: filter chuẩn (train/val theo split)
        train_loader = get_dataloader(
            csv_path=csv_path,
            image_dir=image_dir,
            roi_size=roi_size,
            split="train",
            fold=fold,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            use_fft=use_fft,
            fft_d0=fft_d0,
            fft_cache_dir=fft_cache_dir,
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
            pin_memory=False,
            use_fft=use_fft,
            fft_d0=fft_d0,
            fft_cache_dir=fft_cache_dir,
        )
    else:
        # Wide-format (holdout): filter theo fold (placeholder split="train" cho pool)
        train_loader, val_loader = _get_wide_format_dataloaders(
            csv_path=csv_path,
            image_dir=image_dir,
            fold=fold,
            roi_size=roi_size,
            batch_size=batch_size,
            num_workers=num_workers,
            use_fft=use_fft,
            fft_d0=fft_d0,
            fft_cache_dir=fft_cache_dir,
        )

    return train_loader, val_loader


def _get_wide_format_dataloaders(
    csv_path: str,
    image_dir: str,
    fold: int,
    roi_size: int,
    batch_size: int,
    num_workers: int,
    use_fft: bool,
    fft_d0: float,
    fft_cache_dir: str = None,
):
    """Wide-format (holdout CSV): filter theo fold thuần (split="train" placeholder)."""
    from torch.utils.data import DataLoader, Dataset

    df = pd.read_csv(csv_path)
    pool_df = df[df["fold"].between(1, 10)].copy()

    train_df = pool_df[pool_df["fold"] != fold]
    val_df = pool_df[pool_df["fold"] == fold]

    class _WideDataset(Dataset):
        def __init__(self, df, image_dir, roi_size, transform, use_fft, fft_d0, fft_cache_dir):
            self.df = df.reset_index(drop=True)
            self.image_dir = image_dir
            self.roi_size = roi_size
            self.transform = transform
            self.use_fft = use_fft
            self.fft_d0 = fft_d0
            self.fft_cache_dir = fft_cache_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, row["image_idx"])
            if self.use_fft and self.fft_cache_dir:
                cached_path = os.path.join(self.fft_cache_dir, row["image_idx"])
                if os.path.exists(cached_path):
                    image = Image.open(cached_path).convert("RGB")
                else:
                    image = Image.open(img_path).convert("RGB")
                    image = center_crop_roi(image, self.roi_size)
                    if self.use_fft:
                        image = fft_lowpass_denoise_pil(image, d0=self.fft_d0)
            else:
                image = Image.open(img_path).convert("RGB")
                image = center_crop_roi(image, self.roi_size)
                if self.use_fft:
                    image = fft_lowpass_denoise_pil(image, d0=self.fft_d0)

            label = torch.tensor(row["blood(mg/dL)"], dtype=torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, label

    print(
        f"[WideDataset/train] Loaded {len(train_df)} samples | roi_size={roi_size} | "
        f"fold ∈ [1,10] \\ {fold}"
    )
    print(
        f"[WideDataset/val]   Loaded {len(val_df)} samples | roi_size={roi_size} | fold={fold}"
    )

    train_dataset = _WideDataset(
        train_df, image_dir, roi_size, get_train_transform(roi_size),
        use_fft, fft_d0, fft_cache_dir,
    )
    val_dataset = _WideDataset(
        val_df, image_dir, roi_size, get_valid_transform(roi_size),
        use_fft, fft_d0, fft_cache_dir,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )
    return train_loader, val_loader


def get_pool_train_loader(
    csv_path: str,
    image_dir: str,
    roi_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_fft: bool = False,
    fft_d0: float = 30.0,
    fft_cache_dir: str = None,
):
    """
    Holdout workflow: trả về DataLoader gộp TOÀN BỘ pool rows
    (fold ∈ {1..10}, split="train" placeholder) để train final model.

    Không filter theo fold — dùng toàn bộ 80% pool làm training set.
    """
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    df_full = pd.read_csv(csv_path)
    pool_df = df_full[df_full["fold"].between(1, 10)].copy()

    class _PoolDataset(Dataset):
        def __init__(self, df, image_dir, roi_size, transform, use_fft, fft_d0, fft_cache_dir):
            self.df = df.reset_index(drop=True)
            self.image_dir = image_dir
            self.roi_size = roi_size
            self.transform = transform
            self.use_fft = use_fft
            self.fft_d0 = fft_d0
            self.fft_cache_dir = fft_cache_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, row["image_idx"])

            if self.use_fft and self.fft_cache_dir:
                cached_path = os.path.join(self.fft_cache_dir, row["image_idx"])
                if os.path.exists(cached_path):
                    image = Image.open(cached_path).convert("RGB")
                else:
                    image = Image.open(img_path).convert("RGB")
                    image = center_crop_roi(image, self.roi_size)
                    if self.use_fft:
                        image = fft_lowpass_denoise_pil(image, d0=self.fft_d0)
            else:
                image = Image.open(img_path).convert("RGB")
                image = center_crop_roi(image, self.roi_size)
                if self.use_fft:
                    image = fft_lowpass_denoise_pil(image, d0=self.fft_d0)

            label = torch.tensor(row["blood(mg/dL)"], dtype=torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, label

    print(
        f"[PoolDataset] Loaded {len(pool_df)} samples | roi_size={roi_size} | "
        f"pool rows (fold ∈ [1, {pool_df['fold'].max() if len(pool_df) else 10}])"
    )

    dataset = _PoolDataset(
        df=pool_df,
        image_dir=image_dir,
        roi_size=roi_size,
        transform=get_train_transform(roi_size),
        use_fft=use_fft,
        fft_d0=fft_d0,
        fft_cache_dir=fft_cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )


def get_test_fixed_loader(
    csv_path: str,
    image_dir: str,
    roi_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_fft: bool = False,
    fft_d0: float = 30.0,
    fft_cache_dir: str = None,
):
    """
    Holdout workflow: trả về DataLoader cho test_fixed (fold=0, split=test_fixed).
    """
    return get_dataloader(
        csv_path=csv_path,
        image_dir=image_dir,
        roi_size=roi_size,
        fold=0,
        split="test_fixed",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        use_fft=use_fft,
        fft_d0=fft_d0,
        fft_cache_dir=fft_cache_dir,
    )


def get_fold_dataset_info(csv_path: str, fold: int, splits=("train", "val", "test")):
    """
    Trả về dict thông tin dataset của 1 fold: số patient, số images per split.

    Auto-detect format:
    - Long-format: filter split chuẩn (train/val/test)
    - Wide-format (holdout): chỉ xem fold N là val, các fold khác là train (không có test
      xoay vòng trong pool)
    """
    df = pd.read_csv(csv_path)
    fold_df = df[df["fold"] == fold].copy()

    has_long_format = (
        "split" in fold_df.columns
        and (fold_df["split"] == "val").any()
    )

    info = {"fold": fold}
    if has_long_format:
        for split in splits:
            s = fold_df[fold_df["split"] == split]
            info[f"{split}_patients"] = s["patient_id"].nunique()
            info[f"{split}_images"] = len(s)
            if "blood(mg/dL)" in s.columns:
                labels = s["blood(mg/dL)"].dropna()
                if len(labels):
                    info[f"{split}_label_mean"] = round(labels.mean(), 2)
                    info[f"{split}_label_std"] = round(labels.std(), 2)
    else:
        # Wide-format holdout: 2 splits thô — train (khác fold) và val (== fold)
        pool_df = df[df["fold"].between(1, 10)].copy()
        train_df = pool_df[pool_df["fold"] != fold]
        val_df = pool_df[pool_df["fold"] == fold]
        for split_name, sub in [("train", train_df), ("val", val_df)]:
            info[f"{split_name}_patients"] = sub["patient_id"].nunique()
            info[f"{split_name}_images"] = len(sub)
            if "blood(mg/dL)" in sub.columns and len(sub):
                labels = sub["blood(mg/dL)"].dropna()
                info[f"{split_name}_label_mean"] = round(labels.mean(), 2)
                info[f"{split_name}_label_std"] = round(labels.std(), 2)

    return info


def save_roi_for_fold(
    csv_path: str,
    image_dir: str,
    fold: int,
    roi_size: int,
    output_root: str,
    splits=("train", "val", "test"),
    use_fft: bool = False,
    fft_d0: float = 30.0,
):
    """
    Lưu ROI (center crop) của từng fold ra disk để kiểm tra trực quan.
    Nếu use_fft=True, áp dụng FFT Gaussian low-pass trước khi lưu.

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
            if use_fft:
                roi = fft_lowpass_denoise_pil(roi, d0=fft_d0)
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
    if os.path.exists(all_manifest_path) and os.path.getsize(all_manifest_path) > 0:
        prev = pd.read_csv(all_manifest_path)
        prev = prev[prev["fold"] != fold]
        manifest = pd.concat([prev, manifest], ignore_index=True)
    manifest.to_csv(all_manifest_path, index=False)

    return fold_df, fold_info


def save_roi_for_holdout_fold(
    csv_path: str,
    image_dir: str,
    fold: int,
    roi_size: int,
    output_root: str,
    use_fft: bool = False,
    fft_d0: float = 30.0,
):
    """
    Wide-format holdout: lưu ROI cho 2 splits (train và val) theo fold N.

    Logic wide-format (split_holdout.csv):
      - Train: pool_df[fold ∈ {1..10} AND fold != N]  (9 folds khác)
      - Val:   pool_df[fold == N]                    (1 fold)

    Skip nếu file đã tồn tại (idempotent).

    Cấu trúc:
        output_root/
            foldXX_holdout/
                train/  *.png
                val/    *.png
                foldXX_holdout_roi_manifest.csv
    """
    from PIL import Image
    from tqdm import tqdm

    df = pd.read_csv(csv_path)
    pool_df = df[df["fold"].between(1, 10)].copy()

    train_df = pool_df[pool_df["fold"] != fold].copy()
    val_df = pool_df[pool_df["fold"] == fold].copy()

    fold_dir = os.path.join(output_root, f"fold{fold:02d}_holdout")
    saved_rows = []

    splits_data = [("train", train_df), ("val", val_df)]
    for split_name, sub_df in splits_data:
        split_dir = os.path.join(fold_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for _, row in tqdm(sub_df.iterrows(), desc=f"[ROI holdout] fold{fold:02d}/{split_name}", leave=False):
            img_path = os.path.join(image_dir, row["image_idx"])
            dst_path = os.path.join(split_dir, os.path.basename(row["image_idx"]))

            if os.path.exists(dst_path):
                saved_rows.append({
                    "fold": fold,
                    "split": split_name,
                    "image_idx": row["image_idx"],
                    "roi_path": dst_path,
                    "label": row.get("blood(mg/dL)", None),
                })
                continue
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            roi = center_crop_roi(image, roi_size)
            if use_fft:
                roi = fft_lowpass_denoise_pil(roi, d0=fft_d0)
            roi.save(dst_path)

            saved_rows.append({
                "fold": fold,
                "split": split_name,
                "image_idx": row["image_idx"],
                "roi_path": dst_path,
                "label": row.get("blood(mg/dL)", None),
            })

    manifest = pd.DataFrame(saved_rows)
    manifest_path = os.path.join(fold_dir, f"fold{fold:02d}_holdout_roi_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    # Stats trả về giống get_fold_dataset_info
    info = {
        "fold": fold,
        "train_patients": train_df["patient_id"].nunique(),
        "train_images": len(train_df),
        "val_patients": val_df["patient_id"].nunique(),
        "val_images": len(val_df),
    }
    if "blood(mg/dL)" in train_df.columns:
        info["train_label_mean"] = round(train_df["blood(mg/dL)"].mean(), 2)
        info["train_label_std"] = round(train_df["blood(mg/dL)"].std(), 2)
        info["val_label_mean"] = round(val_df["blood(mg/dL)"].mean(), 2)
        info["val_label_std"] = round(val_df["blood(mg/dL)"].std(), 2)

    return fold_df_passthrough(df, fold), info


def fold_df_passthrough(df, fold):
    """Helper: trả về DataFrame của 1 fold (giữ API tương thích)."""
    return df[df["fold"] == fold].copy()


def precompute_fft_cache(
    csv_path: str,
    image_dir: str,
    cache_dir: str,
    roi_size: int,
    fft_d0: float = 30.0,
    n_folds: int = 10,
):
    """
    Precompute ROI đã áp dụng FFT Low-pass cho tất cả ảnh trong CSV,
    lưu vào cache_dir/<image_idx>.

    Skip nếu file cache đã tồn tại (idempotent, an toàn chạy lại).
    Trả về dict thống kê: {n_total, n_skipped, n_processed, n_missing}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    # Lấy unique image_idx để tránh trùng
    unique_imgs = df["image_idx"].drop_duplicates().tolist()

    n_total = len(unique_imgs)
    n_skipped = 0
    n_processed = 0
    n_missing = 0

    for img_idx in tqdm(unique_imgs, desc=f"[FFT Cache d0={fft_d0}]", leave=False):
        dst = os.path.join(cache_dir, img_idx)
        if os.path.exists(dst):
            n_skipped += 1
            continue
        src = os.path.join(image_dir, img_idx)
        if not os.path.exists(src):
            n_missing += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
            roi = center_crop_roi(img, roi_size)
            roi_fft = fft_lowpass_denoise_pil(roi, d0=fft_d0)
            roi_fft.save(dst)
            n_processed += 1
        except Exception as e:
            print(f"[WARN] Failed processing {img_idx}: {e}")
            n_missing += 1

    return {
        "n_total": n_total,
        "n_skipped": n_skipped,
        "n_processed": n_processed,
        "n_missing": n_missing,
        "cache_dir": cache_dir,
    }
