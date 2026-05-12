#!/usr/bin/env bash
# Phase 1B: 진단 검증.
# 여러 데이터 소스로 같은 모델 진단해서 결과 일관성 확인.
#
# 사용:
#   bash launch_phase1_validate.sh
#
# 시간: ~15분 (3 소스 × 60 params)

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_BATCHES="${NUM_BATCHES:-8}"   # 이전 4 → 8 (더 안정적인 그래디언트)

# 데이터 소스 (random은 비교군, fineweb/wikipedia가 진짜)
SOURCES="${SOURCES:-random_tokens fineweb wikipedia}"

# 파라미터 수 줄여서 시간 단축 (이전 169개 → 60개씩 × 3 소스)
MAX_PARAMS="${MAX_PARAMS:-60}"

# 핵심 (k, α) 조합만
KS="${KS:-8 32}"
ALPHAS="${ALPHAS:-0.5 0.25 0.125}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_TAG=$(basename "$MODEL")
OUTPUT="${OUTPUT:-results/phase1_validate_${MODEL_TAG}.json}"

echo "=== Phase 1B: Validation Across Data Sources ==="
echo "Model:       $MODEL"
echo "Sources:     $SOURCES"
echo "Batches:     $NUM_BATCHES per source"
echo "Max params:  $MAX_PARAMS per source"
echo "Output:      $OUTPUT"
echo ""

python scripts/phase1_validate.py \
    --model_name "$MODEL" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --num_batches $NUM_BATCHES \
    --sources $SOURCES \
    --ks $KS \
    --alphas $ALPHAS \
    --cert_threshold 0.05 \
    --dtype bfloat16 \
    --max_params $MAX_PARAMS \
    --output "$OUTPUT"

echo ""
echo "=== Done. Review the VERDICT section above. ==="
