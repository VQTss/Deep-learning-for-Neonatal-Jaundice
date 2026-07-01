#!/bin/bash
# ==============================================================================
# Train all models for Neonatal Jaundice Regression
# Trains both pretrained and scratch (no-pretrain) versions
# ==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# Config
MODEL_LIST=("resnet18" "convnext_tiny" "efficientnet_b0" "efficientnet_b3" "mobilenetv3_small")
EPOCHS=100
LOSS="mse"
SCHEDULER="plateau"
LR="0.001"
PRETRAIN_MODES=("true" "false")  # train cả 2 version

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra MODEL_LIST <<< "$2"
            shift 2
            ;;
        --pretrain)
            # "both", "pretrained", "scratch"
            case "$2" in
                both)
                    PRETRAIN_MODES=("true" "false")
                    ;;
                pretrained)
                    PRETRAIN_MODES=("true")
                    ;;
                scratch)
                    PRETRAIN_MODES=("false")
                    ;;
                *)
                    echo "Unknown pretrain mode: $2. Use: both, pretrained, scratch"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --epochs N       Number of epochs (default: $EPOCHS)"
            echo "  --loss NAME      Loss function: mse, smoothl1, l1 (default: $LOSS)"
            echo "  --scheduler NAME Scheduler: plateau, cosine, step (default: $SCHEDULER)"
            echo "  --lr FLOAT      Learning rate (default: $LR)"
            echo "  --models LIST   Comma-separated model list (default: all)"
            echo "  --pretrain MODE  Pretrain mode: both, pretrained, scratch (default: both)"
            echo ""
            echo "Available models: ${MODEL_LIST[*]}"
            echo "Output structure:"
            echo "  Pretrained: checkpoint/Regression/{model}_pretrain/"
            echo "  Scratch:    checkpoint/Regression/{model}_scratch/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  Training All Models - Neonatal Jaundice Regression"
echo "============================================================"
echo "  Epochs:     $EPOCHS"
echo "  Loss:       $LOSS"
echo "  Scheduler:  $SCHEDULER"
echo "  LR:         $LR"
echo "  Models:     ${MODEL_LIST[*]}"
echo "  Pretrain:   ${PRETRAIN_MODES[*]}"
echo "============================================================"

START_TIME=$(date +%s)
TOTAL_JOBS=$((${#MODEL_LIST[@]} * ${#PRETRAIN_MODES[@]}))
CURRENT_JOB=0

for MODEL in "${MODEL_LIST[@]}"; do
    for PRETRAIN in "${PRETRAIN_MODES[@]}"; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        TAG="${MODEL}_$( [ "$PRETRAIN" = "true" ] && echo 'pretrain' || echo 'scratch' )"

        echo ""
        echo "============================================================"
        echo "  [$CURRENT_JOB/$TOTAL_JOBS] Starting training: $TAG"
        echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"

        MODEL_START=$(date +%s)

        python train.py \
            --model_name "$MODEL" \
            --epochs "$EPOCHS" \
            --loss_name "$LOSS" \
            --scheduler_name "$SCHEDULER" \
            --lr "$LR" \
            --pretrained "$PRETRAIN"

        MODEL_END=$(date +%s)
        MODEL_DURATION=$((MODEL_END - MODEL_START))
        MODEL_HOURS=$((MODEL_DURATION / 3600))
        MODEL_MINS=$(((MODEL_DURATION % 3600) / 60))
        MODEL_SECS=$((MODEL_DURATION % 60))

        echo ""
        echo "============================================================"
        echo "  Completed: $TAG"
        echo "  Duration: ${MODEL_HOURS}h ${MODEL_MINS}m ${MODEL_SECS}s"
        echo "============================================================"
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECS=$((TOTAL_DURATION % 60))

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETED"
echo "  Total jobs: $TOTAL_JOBS"
echo "  Total duration: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
