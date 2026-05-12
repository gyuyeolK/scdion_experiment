#!/usr/bin/env bash
# Phase 2 (small-scale): 360M 모델로 빠른 wall-clock 비교.
#
# 사용법:
#   bash launch_phase2_small.sh muon
#   bash launch_phase2_small.sh dion2_uniform
#   bash launch_phase2_small.sh sc_dion
#
# 시간: 각 15-20분, 총 ~1시간
#
# 끝나면:
#   python analysis/analyze_results.py runs_small/muon runs_small/dion2_uniform_a0.5 runs_small/sc_dion_a0.5

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

OPTIMIZER="${1:-muon}"

# 360M 모델로 빠르게
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-360M}"

# 짧은 학습
MAX_STEPS="${MAX_STEPS:-500}"
WARMUP="${WARMUP:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"

# Conservative learning rate (continued pretraining)
LR="${LR:-5e-4}"
LR_MIN="${LR_MIN:-5e-5}"

# Optimizer params
ALPHA="${ALPHA:-0.5}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
CERT_THRESHOLD="${CERT_THRESHOLD:-0.05}"
REFRESH_PERIOD="${REFRESH_PERIOD:-20}"
SELECTOR="${SELECTOR:-topk}"   # topk (fast), block_greedy, greedy

# Seed (모든 옵티마이저에 같은 seed 사용 → 같은 데이터 순서)
SEED="${SEED:-42}"

# Output
TAG="${TAG:-${OPTIMIZER}_a${ALPHA}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_small/$TAG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== Phase 2 (small): Quick wall-clock comparison ==="
echo "Optimizer:  $OPTIMIZER (α=$ALPHA)"
echo "Model:      $MODEL"
echo "Steps:      $MAX_STEPS (warmup $WARMUP)"
echo "Batch:      $BATCH_SIZE × seq $SEQ_LEN"
echo "Seed:       $SEED"
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
    --eval_interval 50 \
    --log_interval 25

echo ""
echo "=== Done. ==="
echo ""
echo "결과: $OUTPUT_DIR/history.json"
echo ""
echo "셋 다 끝나면:"
echo "  python analysis/analyze_results.py runs_small/muon_a0.5 runs_small/dion2_uniform_a0.5 runs_small/sc_dion_a0.5"
