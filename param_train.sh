#!/bin/bash
# filepath: train.sh
# Launch script for JITrans Image Classifier (encoder-decoder with learnable y_query)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
ENV_NAME="jit"
DATA_PATH="../datahome/RSCD_dataset_1million_reorganized/"
BATCH_SIZE=64
IMG_SIZE=224
PATCH_SIZE=16
NUM_EPOCHS=200
LR=0.0001
WEIGHT_DECAY=0.0001
#OUTPUT_DIR=""
#OUTPUT_DIR="./output/jitrans_20260615_155632_1gpu"
NUM_WORKERS=""
MULTI_GPU=false
ENABLE_AMP=false
USE_CHECKPOINT=false
DRY_RUN=false

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
show_help() {
    cat << EOF
JITrans Image Classifier — Training Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENV_NAME          Conda environment name (default: base)
    -d, --data DATA_PATH        Path to dataset root (must contain train/val/test)
    -b, --batch BATCH_SIZE      Batch size per GPU (default: 64)
    -i, --img-size SIZE         Input image size, H=W (default: 224)
    -p, --patch-size SIZE       Patch size for image tokenisation (default: 1)
                                Note: in_dim_enc = C * patch_size^2 must match
                                the value set in model.py define_model()
    -n, --epochs N              Number of training epochs (default: 150)
    -l, --lr RATE               Learning rate (default: 1e-4)
    -r, --weight-decay RATE     Weight decay (default: 1e-4)
    -o, --output OUTPUT_DIR     Output directory (default: auto-generated)
    -w, --workers N             DataLoader workers (default: 4 x num_gpus)
    -g, --multi-gpu             Use all available GPUs (torchrun DDP)
    -a, --amp                   Enable Automatic Mixed Precision
    -c, --checkpoint            Enable gradient checkpointing (saves VRAM)
        --dry-run               Show configuration without training
    -h, --help                  Show this help message

EXAMPLES:
    $0                                        # All defaults
    $0 -b 64 -n 100                          # Larger batch, more epochs
    $0 -d /data/RSCD -a -c                   # AMP + gradient checkpointing
    $0 -g -b 32                              # Multi-GPU DDP
    $0 --dry-run                             # Preview config only

GPU DETECTION:
    Single-GPU mode: selects the GPU with the most free VRAM.
    Multi-GPU mode:  uses all available GPUs via torchrun.
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)           ENV_NAME="$2";       shift 2 ;;
        -d|--data)          DATA_PATH="$2";      shift 2 ;;
        -b|--batch)         BATCH_SIZE="$2";     shift 2 ;;
        -i|--img-size)      IMG_SIZE="$2";       shift 2 ;;
        -p|--patch-size)    PATCH_SIZE="$2";     shift 2 ;;
        -n|--epochs)        NUM_EPOCHS="$2";     shift 2 ;;
        -l|--lr)            LR="$2";             shift 2 ;;
        -r|--weight-decay)  WEIGHT_DECAY="$2";   shift 2 ;;
        -o|--output)        OUTPUT_DIR="$2";     shift 2 ;;
        -w|--workers)       NUM_WORKERS="$2";    shift 2 ;;
        -g|--multi-gpu)     MULTI_GPU=true;      shift ;;
        -a|--amp)           ENABLE_AMP=true;     shift ;;
        -c|--checkpoint)    USE_CHECKPOINT=true; shift ;;
        --dry-run)          DRY_RUN=true;        shift ;;
        -h|--help)          show_help; exit 0 ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
detect_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Is CUDA installed?"
        exit 1
    fi

    echo "=== GPU Detection ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
               --format=csv,noheader,nounits | \
    while IFS=, read -r idx name total free; do
        total_gb=$((total / 1024))
        free_gb=$((free / 1024))
        echo "  GPU $idx: $(echo $name | xargs)  [${total_gb} GB total | ${free_gb} GB free]"
    done
    echo ""

    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "ERROR: No GPUs detected!"
        exit 1
    fi

    if [ "$MULTI_GPU" = true ]; then
        GPU_LIST=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        echo "  Mode : Multi-GPU (torchrun DDP)"
        echo "  GPUs : $GPU_LIST  ($GPU_COUNT total)"
    else
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free \
                   --format=csv,noheader,nounits | \
                   sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free \
                   --format=csv,noheader,nounits -i "$BEST_GPU")
        FREE_GB=$((FREE_MEM / 1024))
        echo "  Mode : Single-GPU (auto-selected)"
        echo "  Using GPU $BEST_GPU  (${FREE_GB} GB free)"
        GPU_LIST="$BEST_GPU"
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
        GPU_COUNT=1
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------
calculate_params() {
    [ -z "$NUM_WORKERS" ] && NUM_WORKERS=$((8 * GPU_COUNT))
    TOTAL_BATCH=$((BATCH_SIZE * GPU_COUNT))

    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="./output/jitrans_$(date +%Y%m%d_%H%M%S)_${GPU_COUNT}gpu"
    fi

    CHECKPOINT_FLAG=""
    [ "$USE_CHECKPOINT" = true ] && CHECKPOINT_FLAG="--use_checkpoint"

    AMP_FLAG=""
    [ "$ENABLE_AMP" = true ] && AMP_FLAG="--amp"
}

# ---------------------------------------------------------------------------
# Show configuration
# ---------------------------------------------------------------------------
show_config() {
    echo "=== Training Configuration ==="
    echo "  Environment   : $ENV_NAME"
    echo "  Data path     : $DATA_PATH"
    echo "  Image size    : ${IMG_SIZE}x${IMG_SIZE}"
    echo "  Patch size    : ${PATCH_SIZE}x${PATCH_SIZE}"
    echo "  Batch/GPU     : $BATCH_SIZE"
    echo "  Total batch   : $TOTAL_BATCH"
    echo "  Epochs        : $NUM_EPOCHS"
    echo "  LR            : $LR"
    echo "  Weight decay  : $WEIGHT_DECAY"
    echo "  Workers       : $NUM_WORKERS"
    echo "  GPUs          : $GPU_COUNT  ($GPU_LIST)"
    echo "  Mixed prec.   : $ENABLE_AMP"
    echo "  Grad ckpt     : $USE_CHECKPOINT"
    echo "  Output dir    : $OUTPUT_DIR"
    echo "=============================="
    echo ""
}

# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------
validate_inputs() {
    if ! conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} \|/${ENV_NAME}$"; then
        echo "ERROR: Conda environment '$ENV_NAME' not found."
        echo "Available environments:"
        conda env list
        exit 1
    fi

    if [ ! -d "$DATA_PATH" ]; then
        echo "ERROR: Data path '$DATA_PATH' does not exist."
        exit 1
    fi

    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
        echo "ERROR: batch size must be a positive integer."
        exit 1
    fi

    if [ $(( IMG_SIZE % PATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: img_size ($IMG_SIZE) must be divisible by patch_size ($PATCH_SIZE)."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "=== JITrans Image Classifier — Training Script ==="
echo ""

detect_gpus
calculate_params
show_config

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: configuration shown above. Exiting without training."
    exit 0
fi

validate_inputs

mkdir -p "$OUTPUT_DIR"

echo "=== Ready to Start Training ==="

echo "=== Starting Training ==="

if [ "$MULTI_GPU" = true ]; then
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
        --nproc_per_node="$GPU_COUNT" \
        main.py \
        --data_path "$DATA_PATH" \
        --data_set "RSCD" \
        --img_size "$IMG_SIZE" \
        --patch_size "$PATCH_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --num_workers "$NUM_WORKERS" \
        --model_save_dir "$OUTPUT_DIR" \
        $CHECKPOINT_FLAG \
        $AMP_FLAG
else
    CUDA_VISIBLE_DEVICES=$GPU_LIST python main.py \
        --data_path "$DATA_PATH" \
        --data_set "RSCD" \
        --img_size "$IMG_SIZE" \
        --patch_size "$PATCH_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --num_workers "$NUM_WORKERS" \
        --model_save_dir "$OUTPUT_DIR" \
        --resume "$OUTPUT_DIR/last.pt" \
        $CHECKPOINT_FLAG \
        $AMP_FLAG
fi

echo ""
echo "=== Training Completed ==="
echo "  Results saved in : $OUTPUT_DIR"
echo "  Finished at      : $(date)"
echo "  GPUs used        : $GPU_COUNT  ($GPU_LIST)"
echo "  Total batch size : $TOTAL_BATCH"
