import os
import time
import random
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

def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
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


# ==================== Training Loop ====================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp: bool = True):
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
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
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
    test_loader = get_dataloader(
        csv_path=cfg.data_csv,
        image_dir=cfg.data_image,
        fold=fold,
        split="test",
        roi_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # Load model
    model = get_model(cfg.model_name, pretrained=False)
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
    logger.info(f"Test MSE: {metrics['mse']:.4f}")
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


def test_all_folds(cfg: Config, device: torch.device, logger):
    """Evaluate all folds on their test sets"""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING ALL FOLDS")
    logger.info(f"{'='*60}")

    test_results = []
    for fold in range(1, cfg.n_folds + 1):
        result = test_fold(cfg, fold, device, logger)
        if result:
            test_results.append(result)

    # Summary
    df_results = pd.DataFrame(test_results)
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{df_results.to_string(index=False)}")

    mean_mae = df_results["mae"].mean()
    mean_rmse = df_results["rmse"].mean()
    logger.info(f"\nMean Test MAE: {mean_mae:.4f}")
    logger.info(f"Mean Test RMSE: {mean_rmse:.4f}")

    return test_results


# ==================== Main ====================

def train_fold(
    cfg: Config,
    fold: int,
    device: torch.device,
    logger
):
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
    train_loader, val_loader = get_10fold_dataloaders(
        csv_path=cfg.data_csv,
        image_dir=cfg.data_image,
        fold=fold,
        roi_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    
    logger.info(f"========= Pretrain {cfg.pretrained} =========")

    # Model
    model = get_model(cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device)

    # Loss
    criterion = losses_function(cfg.loss_name, beta=cfg.loss_beta)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

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

    # AMP Scaler
    scaler = GradScaler(device.type) if cfg.use_amp else None

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, cfg.use_amp
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step
        if scheduler is not None:
            if cfg.scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        elapsed = time.time() - start_time

        # Current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        logger.info(
            f"Epoch {epoch}/{cfg.epochs} | LR: {current_lr:.2e} | "
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

    # Output dirs: checkpoint/Regression/{model_name}_{pretrain_mode}/
    pretrain_tag = "pretrain" if cfg.pretrained else "scratch"
    cfg.output_dir = f"checkpoint/Regression/{cfg.model_name}_{pretrain_tag}"
    cfg.checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    cfg.log_dir = os.path.join(cfg.output_dir, "logs")
    cfg.plot_dir = os.path.join(cfg.output_dir, "plots")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)

    # Logger (khởi tạo trước để có thể log LR adjustment)
    log_file = os.path.join(cfg.log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger("train", level="INFO", log_file=log_file)

    # Auto-adjust LR cho từng model family (EfficientNet cần LR thấp hơn)
    if cfg.model_name.startswith("efficientnet"):
        if cfg.lr >= 1e-3:  # Chỉ hạ nếu đang dùng LR mặc định cao
            original_lr = cfg.lr
            cfg.lr = 5e-5   # EfficientNet: LR thấp hơn để ổn định hơn
            logger.info(f"  [Auto] LR adjusted: {original_lr} -> {cfg.lr} (model={cfg.model_name})")

    # Config summary
    logger.info("=" * 60)
    logger.info("CONFIG SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Model:        {cfg.model_name} (pretrained={cfg.pretrained})")
    logger.info(f"  Task:         Regression")
    logger.info(f"  K-fold:       {cfg.n_folds}")
    logger.info(f"  Epochs:       {cfg.epochs}")
    logger.info(f"  Batch size:   {cfg.batch_size}")
    logger.info(f"  LR:           {cfg.lr} (decayed by {cfg.scheduler_name})")
    logger.info(f"  Optimizer:    AdamW (weight_decay={cfg.weight_decay})")
    logger.info(f"  Loss:         {cfg.loss_name}")
    logger.info(f"  Scheduler:    {cfg.scheduler_name}")
    if cfg.scheduler_name == "plateau":
        logger.info(f"    - patience: {cfg.plateau_patience}, factor: {cfg.plateau_factor}, min_lr: {cfg.plateau_min_lr}")
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
