#!/usr/bin/env bash
# Phase 1D: 학습 중 다양한 시점에서 그래디언트 구조 진단.
#
# 사용법:
#   bash launch_phase1_intraining.sh
#
# 시간: ~1.5시간 (1000 step 학습 + 6번 진단)
# - 학습: 1000 step × 1초/step ≈ 17분
# - 진단: 6번 × 7분 ≈ 42분
# - 총: ~1시간

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1.7B로 진행 (이전 진단 결과 있어서 비교 가능)
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"

BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-2048}"

# 진단할 step: 학습 초기, 짧은 warmup 끝, 중기, 후기
DIAGNOSE_STEPS="${DIAGNOSE_STEPS:-0 10 50 200 500 1000}"
MAX_STEPS="${MAX_STEPS:-1000}"

# 시간 절약: 파라미터 60→40개
MAX_PARAMS="${MAX_PARAMS:-40}"

# Learning rate (continued pre-training scale)
LR="${LR:-2e-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_TAG=$(basename "$MODEL")
OUTPUT="${OUTPUT:-results/phase1_intraining_${MODEL_TAG}.json}"

echo "=== Phase 1D: In-Training Diagnostic ==="
echo "Model:         $MODEL"
echo "Train steps:   $MAX_STEPS"
echo "Diagnose at:   $DIAGNOSE_STEPS"
echo "Per diagnostic: $MAX_PARAMS params"
echo "Learning rate: $LR"
echo "Output:        $OUTPUT"
echo ""

python scripts/phase1_intraining.py \
    --model_name "$MODEL" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --diagnose_steps $DIAGNOSE_STEPS \
    --max_steps $MAX_STEPS \
    --max_params $MAX_PARAMS \
    --ks 8 32 \
    --alphas 0.5 0.25 0.125 \
    --cert_thresholds 0.05 0.1 0.2 0.3 \
    --lr $LR \
    --dtype bfloat16 \
    --output "$OUTPUT"

echo ""
echo "=== Done. Review VERDICT above. ==="
echo ""
echo "결과 파일: $OUTPUT"
echo "주요 데이터:"
echo "  - diagnostics[].grad / .momentum: 시점별 그래디언트/모멘텀 구조"
echo "  - losses[]: 전체 loss 곡선 (학습이 잘 됐는지 확인)"
