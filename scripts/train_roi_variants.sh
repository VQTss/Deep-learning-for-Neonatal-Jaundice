#!/usr/bin/env bash
# =============================================================================
#  train_roi_variants.sh
# =============================================================================
#  Chạy 4 ROI variants × convnext_tiny × holdout split.
#  Workflow mỗi variant: 10-fold CV trên 80% pool + 1 final model trên 20% test_fixed.
#
#  Variants:
#    A) ROI 224×224 + FFT Low-pass (d0=30) — dùng cache images_wb_region_fft_d0_30
#    B) ROI 224×224 (no FFT)
#    C) ROI 128×128 + FFT Low-pass (d0=30)
#    D) ROI 128×128 (no FFT)
#
#  Common settings:
#    - dataset:   images_wb_region (ROI 567×567 center-crop từ ảnh gốc)
#    - csv:       split_holdout.csv (CV pool + test_fixed)
#    - model:     convnext_tiny (pretrained)
#    - loss:      smoothl1
#    - epochs:    100
#    - n_folds:   10
#    - batch:     32
#
#  Output: mỗi variant có checkpoint_dir riêng để resume + ranking sau.
#  Log:    logs/<run_tag>.log (auto-rotate theo timestamp)
#
#  NOTE: Color space (RGB+HSV, RGB+LAB, etc. — multibranch) yêu cầu
#  multibranch model + preprocessing riêng — CHƯA implement, xem scope.
# =============================================================================

set -euo pipefail

# ----- Working dirs -----
cd "$(dirname "$0")/.."        # → /home/.../NJ-v5
REPO_ROOT="$(pwd)"
cd src                        # train.py expects working dir = src/

# ----- Common defaults -----
DATA_CSV="split_holdout.csv"
DATA_IMAGE="images_wb_region"            # ROI folder 567×567 (đã crop từ ảnh gốc)
DATA_PATH="../datasets"
SPLIT_MODE="holdout"
MODEL="convnext_tiny"
EPOCHS=100
N_FOLDS=10
BATCH=32
FFT_D0=30
SEED=42
OUTPUT_BASE="../checkpoint_v3"

# ----- Resume flag (set non-empty để skip khi đã có result JSON) -----
RESUME_FLAG=""  # để trống để train từ đầu; set "RESUME=1" trong env để resume

mkdir -p "${OUTPUT_BASE}" ../logs

run_one_variant() {
    local roi_size="$1"
    local use_fft="$2"
    local tag="$3"

    local fft_flag="false"
    [[ "${use_fft}" == "1" ]] && fft_flag="true"

    local tag_fft
    if [[ "${use_fft}" == "1" ]]; then
        tag_fft="fft_d${FFT_D0}"
    else
        tag_fft="nofft"
    fi

    local run_root="${OUTPUT_BASE}"
    local run_tag="roi${roi_size}_${tag_fft}_${MODEL}_holdout"
    local log_file="../logs/${run_tag}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo " Variant: ${tag}"
    echo "   - ROI:        ${roi_size}×${roi_size}"
    echo "   - FFT Low-pass: ${tag_fft}"
    echo "   - Model:       ${MODEL}"
    echo "   - Split mode:  ${SPLIT_MODE}"
    echo "   - Folds:       ${N_FOLDS}"
    echo "   - Epochs:      ${EPOCHS}"
    echo "   - Output root: ${run_root}"
    echo "   - Run tag:     ${run_tag}"
    echo "   - Log file:    ${log_file}"
    echo "================================================================"

    python train.py \
        --model_name "${MODEL}" \
        --pretrained true \
        --loss_name smoothl1 \
        --loss_beta 1.0 \
        --scheduler_name plateau \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH}" \
        --early_stopping true \
        --patience 10 \
        --warmup_epochs 3 \
        --lr 1e-4 \
        --lr_backbone 5e-5 \
        --lr_head 5e-4 \
        --weight_decay 1e-4 \
        --use_amp true \
        --data_path "${DATA_PATH}" \
        --data_csv "${DATA_CSV}" \
        --data_image "${DATA_IMAGE}" \
        --image_size "${roi_size}" \
        --n_folds "${N_FOLDS}" \
        --split_mode "${SPLIT_MODE}" \
        --use_fft "${fft_flag}" \
        --fft_d0 "${FFT_D0}" \
        --seed "${SEED}" \
        --output_root "${run_root}" \
        --run_tag "${run_tag}" \
        2>&1 | tee "${log_file}"
}

# =============================================================================
#  Chạy 4 ROI variants
# =============================================================================

# A) ROI 224 + FFT Low-pass
run_one_variant 224 1 "ROI 224x224 + FFT (d0=30)"

# B) ROI 224 không FFT
run_one_variant 224 0 "ROI 224x224 (no FFT)"

# C) ROI 128 + FFT Low-pass
run_one_variant 128 1 "ROI 128x128 + FFT (d0=30)"

# D) ROI 128 không FFT
run_one_variant 128 0 "ROI 128x128 (no FFT)"

echo ""
echo "================================================================"
echo "  DONE. Tổng cộng 4 ROI variants đã train xong."
echo "  Output root: ${OUTPUT_BASE}/"
echo "    roi224_fft_d${FFT_D0}_convnext_tiny_holdout/   (ROI 224 × FFT)"
echo "    roi224_nofft_convnext_tiny_holdout/            (ROI 224 × no FFT)"
echo "    roi128_fft_d${FFT_D0}_convnext_tiny_holdout/   (ROI 128 × FFT)"
echo "    roi128_nofft_convnext_tiny_holdout/            (ROI 128 × no FFT)"
echo "  Mỗi output chứa:"
echo "    - fold01_result.json ... fold10_result.json (CV per fold)"
echo "    - final_holdout_result.json (final model trên test_fixed)"
echo "    - logs/ (training log per fold)"
echo "    - plots/ (loss curve per fold)"
echo "    - checkpoints/fold{N}_best.pth, checkpoints/final_holdout.pth"
echo "  Logs: logs/roi*_convnext_tiny_holdout_*.log"
echo "================================================================"
