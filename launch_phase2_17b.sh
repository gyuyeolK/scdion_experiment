#!/usr/bin/env bash
# Phase 2 (1.7B): Wall-clock comparison at scale.
#
# 사용법:
#   bash launch_phase2_17b.sh muon
#   bash launch_phase2_17b.sh dion2_uniform
#   bash launch_phase2_17b.sh sc_dion
#
# 시간: 각 ~30-60분 (1.7B는 360M의 ~3-4배 step time), 총 ~2-3시간

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

OPTIMIZER="${1:-muon}"

# 1.7B
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"

# 1.7B는 step time 더 크므로 step 수 줄임 (충분한 비교)
MAX_STEPS="${MAX_STEPS:-300}"
WARMUP="${WARMUP:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"   # 1.7B는 메모리 큼
SEQ_LEN="${SEQ_LEN:-2048}"

# Continued pretraining LR
LR="${LR:-3e-4}"
LR_MIN="${LR_MIN:-3e-5}"

# Optimizer params
ALPHA="${ALPHA:-0.5}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
CERT_THRESHOLD="${CERT_THRESHOLD:-0.05}"
REFRESH_PERIOD="${REFRESH_PERIOD:-20}"
SELECTOR="${SELECTOR:-topk}"   # 1.7B에서는 빠른 selector 권장

SEED="${SEED:-42}"

TAG="${TAG:-${OPTIMIZER}_a${ALPHA}_${SELECTOR}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_17b/$TAG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== Phase 2 (1.7B): Wall-clock comparison at scale ==="
echo "Optimizer:  $OPTIMIZER (α=$ALPHA, selector=$SELECTOR)"
echo "Model:      $MODEL"
echo "Steps:      $MAX_STEPS (warmup $WARMUP)"
echo "Batch:      $BATCH_SIZE × seq $SEQ_LEN"
echo "Output:     $OUTPUT_DIR"
echo ""

python scripts/phase2_small.py \
    --optimizer "$OPTIMIZER" \
    --model_name "$MODEL" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP \
    --lr $LR \
    --lr_min $LR_MIN \
    --alpha $ALPHA \
    --subspace_rank $SUBSPACE_RANK \
    --cert_threshold $CERT_THRESHOLD \
    --refresh_period $REFRESH_PERIOD \
    --selector $SELECTOR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --eval_interval 30 \
    --log_interval 10 \
    --fail_fast_step_time_ms 30000

echo ""
echo "=== Done. ==="
echo ""
echo "셋 다 끝나면:"
echo "  python analysis/analyze_phase2_small.py runs_17b/*/"
