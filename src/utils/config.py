from dataclasses import dataclass
import argparse


VALID_COLOR_SPACES = ("rgb", "hsv", "lab", "ycbcr")
VALID_FUSION_MODES = ("single", "multibranch")


@dataclass
class Config:
    # ==================== Model ====================
    model_name: str = "convnext_tiny"     # convnext_tiny, efficientnet_b0, efficientnet_b3, mobilenetv3_small, resnet18, cnn1d
    pretrained: bool = True

    # ==================== Color Space & Fusion ====================
    fusion_mode: str = "single"           # single | multibranch
    color_space: str = "rgb"              # rgb | hsv | lab | ycbcr (used when fusion_mode=single)
    spaces_combo: str = ""                # e.g. "rgb+hsv" (used when fusion_mode=multibranch)

    # ==================== Loss ====================
    loss_name: str = "smoothl1"             # smoothl1 (huber), mse, l1 (mae)
    loss_beta: float = 1.0                # chỉ dùng cho huber/smoothl1

    # ==================== Scheduler ====================
    scheduler_name: str = "plateau"        # cosine, step, multistep, cosine_warm, plateau, none
    T_max: int = 50                       # Dùng cho CosineAnnealing
    eta_min: float = 1e-6                 # Learning rate nhỏ nhất (cho cosine)
    step_size: int = 10                   # Dùng cho StepLR
    gamma: float = 0.1                    # Hệ số giảm learning rate
    # ReduceLROnPlateau
    plateau_patience: int = 3            # epochs không cải thiện trước khi giảm LR
    plateau_factor: float = 0.5           # nhân LR khi plateau
    plateau_min_lr: float = 1e-6          # LR tối thiểu

    # ==================== Training ====================
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    lr_backbone: float = 5e-5             # LR cho backbone khi pretrained=True
    lr_backbone_scratch: float = 1e-4    # LR cho backbone khi pretrained=False
    lr_head: float = 5e-4                 # LR cho lớp FC cuối (luôn train từ đầu)
    weight_decay: float = 1e-4
    num_workers: int = 4
    use_amp: bool = True
    early_stopping: bool = True
    patience: int = 10
    warmup_epochs: int = 3                # warmup thủ công cho optimizer
    grad_clip_norm: float = 1.0           # gradient clipping max_norm

    # ==================== Data ====================
    data_path: str = "../datasets"
    data_csv: str = "split_10fold_blood.csv"           # relative to data_path
    data_image: str = "images_wb_global"               # relative to data_path
    image_size: int = 224
    n_folds: int = 10

    # ==================== Logging & Checkpoint ====================
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    roi_dir: str = "roi_by_fold"           # thư mục lưu ROI để kiểm tra

    # ==================== Misc ====================
    seed: int = 42


def get_config() -> Config:
    parser = argparse.ArgumentParser(description="Regression Training Config")

    # Model
    parser.add_argument("--model_name", type=str, default="convnext_tiny")
    parser.add_argument("--pretrained", type=lambda x: str(x).lower() == "true", default=True)

    # Color space & fusion
    parser.add_argument("--fusion_mode", type=str, default="single", choices=VALID_FUSION_MODES,
                        help="single: train 1 backbone on 1 color space. multibranch: N parallel backbones, concat features.")
    parser.add_argument("--color_space", type=str, default="rgb", choices=VALID_COLOR_SPACES,
                        help="Color space for single-branch mode (ignored when fusion_mode=multibranch).")
    parser.add_argument("--spaces_combo", type=str, default="",
                        help="For multibranch mode: '<space1>+<space2>' (e.g. 'rgb+hsv'). Empty for single.")

    # Loss
    parser.add_argument("--loss_name", type=str, default="smoothl1")
    parser.add_argument("--loss_beta", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--scheduler_name", type=str, default="plateau")
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=1e-6)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_backbone_scratch", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Scheduler (plateau)
    parser.add_argument("--plateau_patience", type=int, default=3)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6)

    # Data
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--data_csv", type=str, default="split_10fold_blood.csv")
    parser.add_argument("--data_image", type=str, default="images_wb_global")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--n_folds", type=int, default=10)

    # Misc
    parser.add_argument("--use_amp", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Tạo Config object
    config = Config(
        model_name=args.model_name,
        pretrained=args.pretrained,
        fusion_mode=args.fusion_mode,
        color_space=args.color_space,
        spaces_combo=args.spaces_combo,
        loss_name=args.loss_name,
        loss_beta=args.loss_beta,
        scheduler_name=args.scheduler_name,
        T_max=args.T_max,
        eta_min=args.eta_min,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        lr_backbone_scratch=args.lr_backbone_scratch,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        grad_clip_norm=args.grad_clip_norm,
        data_path=args.data_path,
        data_csv=args.data_csv,
        data_image=args.data_image,
        image_size=args.image_size,
        n_folds=args.n_folds,
        use_amp=args.use_amp,
        seed=args.seed,
    )

    # Validate color space / fusion mode combo
    if config.fusion_mode == "multibranch":
        if not config.spaces_combo:
            raise ValueError("fusion_mode=multibranch requires --spaces_combo like 'rgb+hsv'")
        parts = config.spaces_combo.split("+")
        if len(parts) != 2:
            raise ValueError(f"--spaces_combo must contain exactly 2 spaces separated by '+', got: {config.spaces_combo}")
        for p in parts:
            if p not in VALID_COLOR_SPACES:
                raise ValueError(f"Invalid color space '{p}' in spaces_combo. Valid: {VALID_COLOR_SPACES}")
        if parts[0] == parts[1]:
            raise ValueError(f"spaces_combo must contain 2 DIFFERENT color spaces, got: {config.spaces_combo}")

    return config


if __name__ == "__main__":
    cfg = get_config()
    print(cfg)