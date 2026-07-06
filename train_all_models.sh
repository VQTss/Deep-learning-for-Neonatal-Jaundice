#!/bin/bash
# ==============================================================================
# run_all.sh — Train all models for Neonatal Jaundice Regression
# Trains both pretrained and scratch (no-pretrain) versions across 5 backbones
#
# Cải tiến so với bản gốc:
#   - Loại bỏ tham số --lr mơ hồ (dễ gây nhầm "chạy MSE" khi tưởng SmoothL1),
#     thay bằng lr_backbone / lr_backbone_scratch / lr_head tường minh
#   - Thêm --plateau_patience, --warmup_epochs để tinh chỉnh scheduler
#     (mặc định patience=6 thay vì 3, tránh giảm LR quá sớm do val set nhỏ)
#   - --keep-going: không dừng cả batch nếu 1 job lỗi (mặc định set -e sẽ dừng)
#   - Mỗi job có log file riêng + master log tổng hợp
#   - Tự động skip job đã hoàn thành (dựa vào fold_metrics.csv cuối cùng)
#   - Tổng hợp kết quả tất cả model vào 1 bảng CSV cuối cùng
#   - Hỗ trợ chọn GPU cụ thể qua --gpu
# ==============================================================================

set -uo pipefail   # KHÔNG dùng set -e mặc định nữa -> xem --keep-going bên dưới

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# ==================== Default Config ====================
MODEL_LIST=("resnet18" "convnext_tiny" "efficientnet_b0" "efficientnet_b3" "mobilenetv3_small")
EPOCHS=100
LOSS="smoothl1"
SCHEDULER="plateau"

# --- FIX: tách rõ LR cho từng nhóm tham số, tránh nhầm lẫn ---
LR_BACKBONE="5e-5"
LR_BACKBONE_SCRATCH="1e-4"
LR_HEAD="5e-4"

# --- FIX: patience mặc định tăng 3 -> 6 (val set nhỏ, patience=3 quá nhạy nhiễu) ---
PLATEAU_PATIENCE=6
PLATEAU_FACTOR=0.5
WARMUP_EPOCHS=3
GRAD_CLIP_NORM=1.0
BATCH_SIZE=32
N_FOLDS=10

# --- Color space & fusion config ---
FUSION_MODES=("single")
COLOR_SPACES=("rgb")
SPACES_COMBOS=("rgb+hsv" "rgb+ycbcr" "rgb+lab" "hsv+lab" "hsv+ycbcr" "lab+ycbcr")

PRETRAIN_MODES=("true" "false")
KEEP_GOING=false
GPU_ID=""
FORCE_RERUN=false

MASTER_LOG="$SCRIPT_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

# ==================== Parse arguments ====================
print_usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --epochs N               Number of epochs (default: $EPOCHS)
  --loss NAME               Loss function: mse, smoothl1, l1 (default: $LOSS)
  --scheduler NAME           Scheduler: plateau, cosine, step (default: $SCHEDULER)
  --lr_backbone FLOAT        LR backbone khi pretrained=True (default: $LR_BACKBONE)
  --lr_backbone_scratch FLOAT LR backbone khi pretrained=False (default: $LR_BACKBONE_SCRATCH)
  --lr_head FLOAT            LR cho FC head (default: $LR_HEAD)
  --plateau_patience N       Patience cho ReduceLROnPlateau (default: $PLATEAU_PATIENCE)
  --plateau_factor FLOAT     Hệ số giảm LR khi plateau (default: $PLATEAU_FACTOR)
  --warmup_epochs N          Số epoch warmup thủ công (default: $WARMUP_EPOCHS)
  --grad_clip_norm FLOAT     Gradient clipping max_norm (default: $GRAD_CLIP_NORM)
  --batch_size N              Batch size (default: $BATCH_SIZE)
  --n_folds N                  Số fold (default: $N_FOLDS)
  --models LIST                Comma-separated model list (default: all)
  --pretrain MODE              Pretrain mode: both, pretrained, scratch (default: both)
  --fusion_modes LIST          Comma-separated fusion modes: single, multibranch (default: single)
  --color_spaces LIST          Comma-separated color spaces (single mode): rgb,hsv,lab,ycbcr (default: rgb)
  --spaces_combos LIST         Comma-separated combos (multibranch mode): rgb+hsv,lab+ycbcr,... (default: all 6 pairs)
  --gpu ID                     Chọn GPU cụ thể (VD: 0). Mặc định dùng GPU mặc định của hệ thống
  --keep-going                 Không dừng batch nếu 1 job lỗi, tiếp tục job tiếp theo
  --force-rerun                Bỏ qua cơ chế skip, chạy lại toàn bộ dù đã có kết quả
  --help                        Hiển thị hướng dẫn này

Available models: ${MODEL_LIST[*]}

Output structure:
  Pretrained:  checkpoint/Regression/{model}_pretrain_[{color_space}|mb_{combo}]/
  Scratch:     checkpoint/Regression/{model}_scratch_[{color_space}|mb_{combo}]/

Examples:
  $0 --models resnet18,convnext_tiny --pretrain both --keep-going
  $0 --plateau_patience 8 --warmup_epochs 5 --gpu 1
  $0 --fusion_modes single,multibranch \
     --color_spaces rgb,hsv,lab,ycbcr \
     --spaces_combos rgb+hsv,rgb+lab,rgb+ycbcr
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --loss) LOSS="$2"; shift 2 ;;
        --scheduler) SCHEDULER="$2"; shift 2 ;;
        --lr_backbone) LR_BACKBONE="$2"; shift 2 ;;
        --lr_backbone_scratch) LR_BACKBONE_SCRATCH="$2"; shift 2 ;;
        --lr_head) LR_HEAD="$2"; shift 2 ;;
        --plateau_patience) PLATEAU_PATIENCE="$2"; shift 2 ;;
        --plateau_factor) PLATEAU_FACTOR="$2"; shift 2 ;;
        --warmup_epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
        --grad_clip_norm) GRAD_CLIP_NORM="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --n_folds) N_FOLDS="$2"; shift 2 ;;
        --models)
            IFS=',' read -ra MODEL_LIST <<< "$2"
            shift 2
            ;;
        --pretrain)
            case "$2" in
                both) PRETRAIN_MODES=("true" "false") ;;
                pretrained) PRETRAIN_MODES=("true") ;;
                scratch) PRETRAIN_MODES=("false") ;;
                *) echo "Unknown pretrain mode: $2. Use: both, pretrained, scratch"; exit 1 ;;
            esac
            shift 2
            ;;
        --fusion_modes)
            IFS=',' read -ra FUSION_MODES <<< "$2"
            shift 2 ;;
        --color_spaces)
            IFS=',' read -ra COLOR_SPACES <<< "$2"
            shift 2 ;;
        --spaces_combos)
            IFS=',' read -ra SPACES_COMBOS <<< "$2"
            shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --keep-going) KEEP_GOING=true; shift ;;
        --force-rerun) FORCE_RERUN=true; shift ;;
        --help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# ==================== Helper: log to both stdout and master log ====================
log() {
    echo "$@" | tee -a "$MASTER_LOG"
}

# ==================== Helper: kiểm tra job đã hoàn thành chưa ====================
# Hỗ trợ cả 2 pattern output dir:
# - Single:      checkpoint/Regression/{model}_{pretrain_tag}_{color_space}
# - Multibranch: checkpoint/Regression/{model}_{pretrain_tag}_mb_{combo_with_underscores}
is_job_done() {
    local model="$1"
    local pretrain_tag="$2"
    local out_dir_single="checkpoint/Regression/${model}_${pretrain_tag}_${3}"
    local out_dir_mb="checkpoint/Regression/${model}_${pretrain_tag}_mb_${3//+/_}"
    local count=0
    for ((f=1; f<=N_FOLDS; f++)); do
        printf -v fold_str "%02d" "$f"
        if [[ -f "${out_dir_single}/fold${fold_str}_result.json" ]] || [[ -f "${out_dir_mb}/fold${fold_str}_result.json" ]]; then
            count=$((count + 1))
        fi
    done
    [[ "$count" -eq "$N_FOLDS" ]]
}

log "============================================================"
log "  Training All Models - Neonatal Jaundice Regression"
log "============================================================"
log "  Epochs:              $EPOCHS"
log "  Loss:                $LOSS"
log "  Scheduler:           $SCHEDULER"
log "  LR backbone:         $LR_BACKBONE (pretrained) / $LR_BACKBONE_SCRATCH (scratch)"
log "  LR head:             $LR_HEAD"
log "  Plateau patience:    $PLATEAU_PATIENCE  factor: $PLATEAU_FACTOR"
log "  Warmup epochs:       $WARMUP_EPOCHS"
log "  Grad clip norm:      $GRAD_CLIP_NORM"
log "  Batch size:          $BATCH_SIZE"
log "  N folds:             $N_FOLDS"
log "  Models:              ${MODEL_LIST[*]}"
log "  Pretrain modes:      ${PRETRAIN_MODES[*]}"
log "  Fusion modes:        ${FUSION_MODES[*]}"
log "  Color spaces:        ${COLOR_SPACES[*]}"
log "  Spaces combos:       ${SPACES_COMBOS[*]}"
log "  GPU:                 ${GPU_ID:-default}"
log "  Keep going on error: $KEEP_GOING"
log "  Force rerun:         $FORCE_RERUN"
log "  Master log:          $MASTER_LOG"
log "============================================================"

START_TIME=$(date +%s)
# Tính tổng số jobs (hỗ trợ nhiều fusion modes)
TOTAL_SS_JOBS=$((${#MODEL_LIST[@]} * ${#PRETRAIN_MODES[@]} * ${#COLOR_SPACES[@]}))
TOTAL_MB_JOBS=$((${#MODEL_LIST[@]} * ${#PRETRAIN_MODES[@]} * ${#SPACES_COMBOS[@]}))
TOTAL_JOBS=$TOTAL_SS_JOBS
for MODE in "${FUSION_MODES[@]}"; do
    if [[ "$MODE" == "multibranch" ]]; then
        TOTAL_JOBS=$((TOTAL_JOBS + TOTAL_MB_JOBS))
    fi
done
CURRENT_JOB=0
FAILED_JOBS=()
SKIPPED_JOBS=()

run_train_job() {
    local model="$1"
    local pretrain="$2"
    local pretrain_tag="$3"
    local cur_lr_backbone="$4"
    local extra_args="$5"      # vd "--color_space lab" hoặc "--fusion_mode multibranch --spaces_combo rgb+hsv"
    local dir_tag="$6"         # vd "lab" hoặc "mb_rgb_hsv" (dùng cho TAG và is_job_done)

    CURRENT_JOB=$((CURRENT_JOB + 1))
    local tag="${model}_${pretrain_tag}_${dir_tag}"

    # --- Skip nếu job đã hoàn thành và không ép force-rerun ---
    if [[ "$FORCE_RERUN" == false ]] && is_job_done "$model" "$pretrain_tag" "$dir_tag"; then
        log ""
        log "[$CURRENT_JOB/$TOTAL_JOBS] SKIP $tag (đã hoàn thành đủ $N_FOLDS fold)"
        SKIPPED_JOBS+=("$tag")
        return
    fi

    local job_log="$SCRIPT_DIR/logs_batch/${tag}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$job_log")"

    log ""
    log "============================================================"
    log "  [$CURRENT_JOB/$TOTAL_JOBS] Starting training: $tag"
    log "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    log "  Job log: $job_log"
    log "============================================================"

    local model_start
    model_start=$(date +%s)

    set +e
    python train.py \
        --model_name "$model" \
        --epochs "$EPOCHS" \
        --loss_name "$LOSS" \
        --scheduler_name "$SCHEDULER" \
        --lr_backbone "$cur_lr_backbone" \
        --lr_backbone_scratch "$LR_BACKBONE_SCRATCH" \
        --lr_head "$LR_HEAD" \
        --plateau_patience "$PLATEAU_PATIENCE" \
        --plateau_factor "$PLATEAU_FACTOR" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --grad_clip_norm "$GRAD_CLIP_NORM" \
        --batch_size "$BATCH_SIZE" \
        --n_folds "$N_FOLDS" \
        --pretrained "$pretrain" \
        $extra_args \
        2>&1 | tee "$job_log"
    local job_exit_code=${PIPESTATUS[0]}
    set -e 2>/dev/null || true

    local model_end model_duration model_hours model_mins model_secs
    model_end=$(date +%s)
    model_duration=$((model_end - model_start))
    model_hours=$((model_duration / 3600))
    model_mins=$(((model_duration % 3600) / 60))
    model_secs=$((model_duration % 60))

    if [[ "$job_exit_code" -ne 0 ]]; then
        log ""
        log "  !! FAILED: $tag (exit code $job_exit_code)"
        log "  Duration: ${model_hours}h ${model_mins}m ${model_secs}s"
        FAILED_JOBS+=("$tag")
        if [[ "$KEEP_GOING" == false ]]; then
            log ""
            log "  Dừng batch (dùng --keep-going để bỏ qua job lỗi và tiếp tục)."
            exit "$job_exit_code"
        fi
    else
        log ""
        log "  Completed: $tag"
        log "  Duration: ${model_hours}h ${model_mins}m ${model_secs}s"
    fi
    log "============================================================"
}

for MODEL in "${MODEL_LIST[@]}"; do
    for PRETRAIN in "${PRETRAIN_MODES[@]}"; do
        PRETRAIN_TAG=$( [ "$PRETRAIN" = "true" ] && echo 'pretrain' || echo 'scratch' )

        # --- Chọn LR backbone phù hợp theo pretrain mode ---
        if [[ "$PRETRAIN" == "true" ]]; then
            CUR_LR_BACKBONE="$LR_BACKBONE"
        else
            CUR_LR_BACKBONE="$LR_BACKBONE_SCRATCH"
        fi

        for MODE in "${FUSION_MODES[@]}"; do
            if [[ "$MODE" == "single" ]]; then
                for SPACE in "${COLOR_SPACES[@]}"; do
                    run_train_job "$MODEL" "$PRETRAIN" "$PRETRAIN_TAG" \
                        "$CUR_LR_BACKBONE" \
                        "--fusion_mode single --color_space $SPACE" \
                        "$SPACE"
                done
            elif [[ "$MODE" == "multibranch" ]]; then
                for COMBO in "${SPACES_COMBOS[@]}"; do
                    # Tag path: combo với '+' -> '_' để tránh shell escape issues
                    COMBO_TAG="${COMBO//+/_}"
                    DIR_TAG="mb_${COMBO_TAG}"
                    run_train_job "$MODEL" "$PRETRAIN" "$PRETRAIN_TAG" \
                        "$CUR_LR_BACKBONE" \
                        "--fusion_mode multibranch --spaces_combo $COMBO" \
                        "$DIR_TAG"
                done
            fi
        done
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECS=$((TOTAL_DURATION % 60))

log ""
log "============================================================"
log "  ALL TRAINING COMPLETED"
log "  Total jobs:     $TOTAL_JOBS"
log "  Skipped (done): ${#SKIPPED_JOBS[@]}  [${SKIPPED_JOBS[*]}]"
log "  Failed:         ${#FAILED_JOBS[@]}  [${FAILED_JOBS[*]}]"
log "  Total duration: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
log "  Finished at:    $(date '+%Y-%m-%d %H:%M:%S')"
log "============================================================"

# ==================== Tổng hợp kết quả tất cả model vào 1 bảng ====================
SUMMARY_CSV="$SCRIPT_DIR/checkpoint/Regression/summary_all_models.csv"
log ""
log "Tổng hợp kết quả -> $SUMMARY_CSV"

python - <<PYEOF
import os, glob, json, re
import pandas as pd

rows = []

model_list = "${MODEL_LIST[*]}".split()
pretrain_modes = "${PRETRAIN_MODES[*]}".split()
fusion_modes = "${FUSION_MODES[*]}".split()
color_spaces = "${COLOR_SPACES[*]}".split()
spaces_combos = "${SPACES_COMBOS[*]}".split()

for model in model_list:
    for pretrain in pretrain_modes:
        tag = "pretrain" if pretrain == "true" else "scratch"

        # Collect candidate output dirs
        candidates = []
        for mode in fusion_modes:
            if mode == "single":
                for space in color_spaces:
                    out_dir = os.path.join("checkpoint", "Regression", f"{model}_{tag}_{space}")
                    candidates.append((out_dir, mode, space, ""))
            elif mode == "multibranch":
                for combo in spaces_combos:
                    combo_underscore = combo.replace("+", "_")
                    out_dir = os.path.join("checkpoint", "Regression", f"{model}_{tag}_mb_{combo_underscore}")
                    candidates.append((out_dir, mode, "", combo))

        for out_dir, mode, space, combo in candidates:
            if not os.path.isdir(out_dir):
                continue
            result_files = sorted(glob.glob(os.path.join(out_dir, "fold*_result.json")))
            if not result_files:
                continue
            val_losses, best_epochs = [], []
            for rf in result_files:
                with open(rf) as f:
                    d = json.load(f)
                val_losses.append(d["val_loss"])
                best_epochs.append(d["best_epoch"])

            rows.append({
                "model": model,
                "pretrain": tag,
                "fusion_mode": mode,
                "color_space": space,
                "combo": combo,
                "n_folds_done": len(result_files),
                "mean_val_loss": sum(val_losses) / len(val_losses) if val_losses else None,
                "mean_best_epoch": sum(best_epochs) / len(best_epochs) if best_epochs else None,
            })

if rows:
    df = pd.DataFrame(rows).sort_values(["model", "fusion_mode", "color_space", "combo"])
    os.makedirs(os.path.dirname("$SUMMARY_CSV"), exist_ok=True)
    df.to_csv("$SUMMARY_CSV", index=False)
    print(df.to_string(index=False))
else:
    print("Không tìm thấy kết quả nào để tổng hợp.")
PYEOF

log ""
log "Xem chi tiết từng job tại: $SCRIPT_DIR/logs_batch/"
log "Master log: $MASTER_LOG"

if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    exit 1
fi