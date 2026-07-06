import os
import time
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from utils.config import Config, get_config
from utils.dataloader import get_10fold_dataloaders, get_dataloader, get_fold_dataset_info, save_roi_for_fold
from utils.losses import losses_function
from utils.scheduler import get_scheduler
from utils.logger import get_logger

# ==================== Model Registry ====================

def get_color_space_args(cfg: Config):
    """Trả về (color_space, spaces) từ cfg, dùng để truyền vào get_dataloader."""
    if cfg.fusion_mode == "multibranch":
        return cfg.color_space, cfg.spaces_combo.split("+")
    return cfg.color_space, None


def get_model(model_name: str, pretrained: bool = True, fusion_mode: str = "single", spaces: list = None) -> nn.Module:
    if fusion_mode == "multibranch":
        from models.multibranch import MultiBranchRegression
        if spaces is None:
            raise ValueError("fusion_mode='multibranch' yêu cầu spaces != None")
        return MultiBranchRegression(spaces=spaces, backbone_name=model_name, pretrained=pretrained)
    if model_name == "resnet18":
        from models.resnet18 import ResNetRegression
        return ResNetRegression(pretrained=pretrained)
    elif model_name == "convnext_tiny":
        from models.convnext_tiny import ConvNeXtTinyRegression
        return ConvNeXtTinyRegression(pretrained=pretrained)
    elif model_name == "efficientnet_b0":
        from models.efficientnet_b0 import EfficientNetB0Regression
        return EfficientNetB0Regression(pretrained=pretrained)
    elif model_name == "efficientnet_b3":
        from models.efficientnet_b3 import EfficientNetB3Regression
        return EfficientNetB3Regression(pretrained=pretrained)
    elif model_name == "mobilenetv3_small":
        from models.mobilenetv3_small import MobileNetV3SmallRegression
        return MobileNetV3SmallRegression(pretrained=pretrained)
    elif model_name == "cnn1d":
        from models.cnn1d import CNN1DRegression
        in_channels = 3  # single-branch RGB/HSV/LAB/YCbCr đều 3 channels
        return CNN1DRegression(in_channels=in_channels, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== Metrics ====================

def compute_metrics(preds, targets):
    """Tính các metrics cho regression"""
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


# ==================== Optimizer & Warmup ====================

def build_optimizer(model, cfg, pretrained: bool):
    """Differential LR cho backbone (features) và head (fc), tách weight decay khỏi BN/bias.

    Hỗ trợ cả single-branch và multi-branch models:
    - Single: backbone là 1 nn.Sequential, head là 1 nn.Linear (tên 'fc.')
    - Multi-branch: backbone là nn.ModuleList[Sequential,...], head là nn.Sequential MLP
      (tham số ở các layer Linear/Conv2d/BN, không có prefix 'fc.' nên dùng heuristic khác).
    """
    lr_backbone = cfg.lr_backbone if pretrained else cfg.lr_backbone_scratch

    backbone_decay, backbone_no_decay, head_params = [], [], []
    is_multibranch = hasattr(model, "backbones") and hasattr(model, "head") and not hasattr(model, "fc")

    if is_multibranch:
        # Multi-branch: duyệt tất cả sub-backbones cho backbone params
        for backbone in model.backbones:
            for name, p in backbone.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 1 or "bn" in name.lower():
                    backbone_no_decay.append(p)
                else:
                    backbone_decay.append(p)
        head_params = [p for p in model.head.parameters() if p.requires_grad]
    else:
        # Single-branch: dùng heuristic 'fc.' prefix như cũ
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("fc."):
                head_params.append(p)
            elif p.ndim == 1 or "bn" in name.lower():
                backbone_no_decay.append(p)
            else:
                backbone_decay.append(p)

    param_groups = [
        {"params": backbone_decay,    "lr": lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": backbone_no_decay,  "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_params,        "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
    ]
    return torch.optim.AdamW(param_groups)


def linear_warmup_lr(optimizer, epoch_idx: int, warmup_epochs: int, base_lrs):
    """Warmup thủ công, tương thích với ReduceLROnPlateau (gọi theo epoch)."""
    if warmup_epochs <= 0 or epoch_idx >= warmup_epochs:
        return
    scale = (epoch_idx + 1) / warmup_epochs
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


# ==================== Training Loop ====================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp: bool = True, grad_clip_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics


def test_fold(cfg: Config, fold: int, device: torch.device, logger):
    """Evaluate model on test set for a specific fold"""
    logger.info(f"{'='*50}")
    logger.info(f"Testing Fold {fold}/{cfg.n_folds}")
    logger.info(f"{'='*50}")

    # Load test data
    cs_color, cs_spaces = get_color_space_args(cfg)
    test_loader = get_dataloader(
        csv_path=cfg.data_csv,
        image_dir=cfg.data_image,
        fold=fold,
        split="test",
        roi_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        color_space=cs_color,
        spaces=cs_spaces,
    )

    # Load model
    model = get_model(cfg.model_name, pretrained=False, fusion_mode=cfg.fusion_mode,
                      spaces=(cfg.spaces_combo.split("+") if cfg.fusion_mode == "multibranch" else None))
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"fold{fold}_best.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")

    # Criterion
    criterion = losses_function(cfg.loss_name, beta=cfg.loss_beta)

    # Test
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Testing Fold {fold}"):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    avg_loss = total_loss / len(test_loader)

    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test MAE: {metrics['mae']:.4f}")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Test R2: {metrics['r2']:.4f}")

    return {
        "fold": fold,
        "test_loss": avg_loss,
        "mae": metrics['mae'],
        "mse": metrics['mse'],
        "rmse": metrics['rmse'],
        "r2": metrics['r2']
    }


# (test_all_folds đã được inline hóa trong main() để test ngay sau từng fold và cleanup CUDA memory.)


# ==================== Main ====================

def train_fold(
    cfg: Config,
    fold: int,
    device: torch.device,
    logger
):
    # Resume: skip fold nếu đã có result JSON (đã train xong lần trước)
    result_path = os.path.join(cfg.output_dir, f"fold{fold:02d}_result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            saved = json.load(f)
        logger.info(f"{'='*50}")
        logger.info(f"Training Fold {fold}/{cfg.n_folds}")
        logger.info(f"{'='*50}")
        logger.info(f"  ↳ Skip fold {fold} (result exists): val_loss={saved['val_loss']:.4f}, best_epoch={saved['best_epoch']}")
        return saved["val_loss"], saved["best_epoch"]

    logger.info(f"{'='*50}")
    logger.info(f"Training Fold {fold}/{cfg.n_folds}")
    logger.info(f"{'='*50}")

    # Save ROI before training for visual QA
    roi_root = getattr(cfg, "roi_dir", "roi_by_fold")
    if not os.path.isabs(roi_root) and hasattr(cfg, "output_dir") and cfg.output_dir:
        roi_root = os.path.join(cfg.output_dir, roi_root)
    _, fold_info = save_roi_for_fold(
        csv_path=cfg.data_csv,
        image_dir=cfg.data_image,
        fold=fold,
        roi_size=cfg.image_size,
        output_root=roi_root,
    )
    logger.info(f"ROI saved for fold {fold:02d} -> {roi_root}/fold{fold:02d}")
    logger.info(
        f"Dataset info - Train: {fold_info['train_patients']} patients, {fold_info['train_images']} images, "
        f"label {fold_info['train_label_mean']:.2f}+-{fold_info['train_label_std']:.2f} mg/dL | "
        f"Val: {fold_info['val_patients']} patients, {fold_info['val_images']} images, "
        f"label {fold_info['val_label_mean']:.2f}+-{fold_info['val_label_std']:.2f} mg/dL | "
        f"Test: {fold_info['test_patients']} patients, {fold_info['test_images']} images, "
        f"label {fold_info['test_label_mean']:.2f}+-{fold_info['test_label_std']:.2f} mg/dL"
    )

    # Data
    cs_color, cs_spaces = get_color_space_args(cfg)
    train_loader, val_loader = get_10fold_dataloaders(
        csv_path=cfg.data_csv,
        image_dir=cfg.data_image,
        fold=fold,
        roi_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        color_space=cs_color,
        spaces=cs_spaces,
    )
    
    logger.info(f"========= Pretrain {cfg.pretrained} =========")

    # Model
    model = get_model(cfg.model_name, pretrained=cfg.pretrained, fusion_mode=cfg.fusion_mode,
                      spaces=(cfg.spaces_combo.split("+") if cfg.fusion_mode == "multibranch" else None))
    model = model.to(device)

    # Loss
    criterion = losses_function(cfg.loss_name, beta=cfg.loss_beta)

    # Optimizer (differential LR: backbone vs head)
    optimizer = build_optimizer(model, cfg, cfg.pretrained)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # Scheduler
    if cfg.scheduler_name == "plateau":
        scheduler_kwargs = {
            "mode": "min",
            "factor": cfg.plateau_factor,
            "patience": cfg.plateau_patience,
            "min_lr": cfg.plateau_min_lr,
        }
    else:
        scheduler_kwargs = {"T_max": cfg.epochs, "eta_min": cfg.eta_min}
    scheduler = get_scheduler(optimizer, cfg.scheduler_name, **scheduler_kwargs)

    # AMP Scaler (guard khi không có CUDA để tránh GradScaler edge-case trên CPU)
    use_amp_runtime = cfg.use_amp and torch.cuda.is_available()
    scaler = GradScaler(device.type) if use_amp_runtime else None

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        # Warmup thủ công (chỉ áp dụng cho các epoch đầu)
        linear_warmup_lr(optimizer, epoch - 1, cfg.warmup_epochs, base_lrs)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            use_amp_runtime, cfg.grad_clip_norm
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step (chỉ sau khi hết warmup)
        if scheduler is not None and epoch > cfg.warmup_epochs:
            if cfg.scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        elapsed = time.time() - start_time

        # Current LR (lấy LR của cả 3 group — đặc biệt hữu ích khi Plateau giảm LR đồng bộ)
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[2]["lr"]

        # Log
        logger.info(
            f"Epoch {epoch}/{cfg.epochs} | LR_bb: {lr_bb:.2e} | LR_head: {lr_head:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f} | Time: {elapsed:.1f}s"
        )

        # Save best & early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_path = os.path.join(cfg.checkpoint_dir, f"fold{fold}_best.pth")
            torch.save({
                "fold": fold,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"]
            }, save_path)
            logger.info(f"  >> Best model saved! Val Loss: {val_loss:.4f}, Val R2: {val_metrics['r2']:.4f}")
        else:
            epochs_no_improve += 1
            if cfg.early_stopping and epochs_no_improve >= cfg.patience:
                logger.info(
                    f"  >> Early stopping triggered! No improvement for {epochs_no_improve} epochs. "
                    f"Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}"
                )
                break

    logger.info(f"Fold {fold} Complete - Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")

    # Lưu result JSON để lần chạy sau có thể skip
    with open(result_path, "w") as f:
        json.dump({"fold": fold, "val_loss": best_val_loss, "best_epoch": best_epoch}, f)

    return best_val_loss, best_epoch


def main():
    # Config (CLI args override defaults)
    cfg = get_config()

    # Resolve paths relative to project root (not cwd which may be src/)
    def resolve_project_path(rel_path: str, base: str = None) -> str:
        """Resolve path from project root, not cwd."""
        # If already absolute, use as-is
        if os.path.isabs(rel_path):
            return rel_path
        # Try relative to train.py location (project root)
        train_py_dir = os.path.dirname(os.path.abspath(__file__))  # src/
        project_root = os.path.dirname(train_py_dir)              # project root
        # If base provided and is absolute, join with base
        if base and os.path.isabs(base):
            resolved = os.path.join(base, rel_path)
            if os.path.exists(resolved):
                return resolved
        # Otherwise try from project root
        resolved = os.path.join(project_root, rel_path)
        if os.path.exists(resolved):
            return resolved
        # Fallback: try relative to cwd
        return os.path.abspath(rel_path)

    # Resolve data_path first, then use it as base for csv/image
    cfg.data_path = resolve_project_path(cfg.data_path)
    cfg.data_csv = resolve_project_path(cfg.data_csv, base=cfg.data_path)
    cfg.data_image = resolve_project_path(cfg.data_image, base=cfg.data_path)

    # Verify data paths tồn tại NGAY SAU khi resolve (fail sớm trước khi tạo output dirs)
    assert os.path.exists(cfg.data_csv), f"CSV not found: {cfg.data_csv}"
    assert os.path.isdir(cfg.data_image), f"Image dir not found: {cfg.data_image}"

    # Output dirs: checkpoint/Regression/{model_name}_{pretrain_mode}[_{color_space}|_mb_{combo}]/
    pretrain_tag = "pretrain" if cfg.pretrained else "scratch"
    if cfg.fusion_mode == "multibranch":
        # Chuẩn hoá combo: thay '+' bằng '_' để tránh shell escape issues
        combo_tag = cfg.spaces_combo.replace("+", "_")
        cfg.output_dir = f"checkpoint/Regression/{cfg.model_name}_{pretrain_tag}_mb_{combo_tag}"
    else:
        cfg.output_dir = f"checkpoint/Regression/{cfg.model_name}_{pretrain_tag}_{cfg.color_space}"
    cfg.checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    cfg.log_dir = os.path.join(cfg.output_dir, "logs")
    cfg.plot_dir = os.path.join(cfg.output_dir, "plots")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)

    # Logger (khởi tạo trước để có thể log)
    log_file = os.path.join(cfg.log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger("train", level="INFO", log_file=log_file)

    # Config summary
    logger.info("=" * 60)
    logger.info("CONFIG SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Model:        {cfg.model_name} (pretrained={cfg.pretrained})")
    logger.info(f"  Fusion mode:  {cfg.fusion_mode}")
    if cfg.fusion_mode == "multibranch":
        logger.info(f"  Spaces combo: {cfg.spaces_combo}")
    else:
        logger.info(f"  Color space:  {cfg.color_space}")
    logger.info(f"  Task:         Regression")
    logger.info(f"  K-fold:       {cfg.n_folds}")
    logger.info(f"  Epochs:       {cfg.epochs}")
    logger.info(f"  Batch size:   {cfg.batch_size}")
    logger.info(f"  LR (backbone): {cfg.lr_backbone} | LR (head): {cfg.lr_head} (decayed by {cfg.scheduler_name})")
    logger.info(f"  Optimizer:    AdamW (weight_decay={cfg.weight_decay})")
    logger.info(f"  Loss:         {cfg.loss_name}")
    logger.info(f"  Scheduler:    {cfg.scheduler_name}")
    if cfg.scheduler_name == "plateau":
        logger.info(f"    - patience: {cfg.plateau_patience}, factor: {cfg.plateau_factor}, min_lr: {cfg.plateau_min_lr}")
    logger.info(f"  Warmup:       {cfg.warmup_epochs} epochs (manual)")
    logger.info(f"  Grad clip:    max_norm={cfg.grad_clip_norm}")
    logger.info(f"  Early stop:   {cfg.early_stopping} (patience={cfg.patience})")
    logger.info(f"  AMP:          {cfg.use_amp}")
    logger.info(f"  Seed:         {cfg.seed}")
    logger.info(f"  Image size:   {cfg.image_size}")
    logger.info(f"  Data CSV:     {cfg.data_csv}")
    logger.info(f"  Data Image:  {cfg.data_image}")
    logger.info(f"  Output dir:   {cfg.output_dir}")
    logger.info("=" * 60)

    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Training & Testing
    fold_results = []
    test_results = []
    for fold in range(1, cfg.n_folds + 1):
        val_loss, best_epoch = train_fold(cfg, fold, device, logger)
        fold_results.append({"fold": fold, "val_loss": val_loss, "best_epoch": best_epoch})

        # Cleanup CUDA memory giữa các fold để tránh crash
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test fold ngay sau khi train xong
        test_result = test_fold(cfg, fold, device, logger)
        if test_result:
            test_results.append(test_result)

    # Summary training
    logger.info(f"\n{'='*50}")
    logger.info("Training Complete!")
    logger.info(f"{'='*50}")

    df_results = pd.DataFrame(fold_results)
    logger.info(f"\n{df_results.to_string(index=False)}")

    mean_val_loss = df_results["val_loss"].mean()
    logger.info(f"\nMean Val Loss: {mean_val_loss:.4f}")

    # Final summary
    if test_results:
        df_test = pd.DataFrame(test_results)
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"\n{df_test.to_string(index=False)}")
        logger.info(f"\nMean Val Loss: {mean_val_loss:.4f}")
        logger.info(f"Mean Test MAE: {df_test['mae'].mean():.4f}")
        logger.info(f"Mean Test RMSE: {df_test['rmse'].mean():.4f}")
        logger.info(f"Mean Test R2: {df_test['r2'].mean():.4f}")
        logger.info(f"{'='*60}")

    logger.info(f"\nOutputs saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
