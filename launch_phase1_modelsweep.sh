#!/usr/bin/env bash
# Phase 1C: 모델 크기별 진단 trend.
#
# 사용법:
#   bash launch_phase1_modelsweep.sh                         # 기본 4개 모델
#   MODELS="X Y Z" bash launch_phase1_modelsweep.sh          # 커스텀
#
# 시간: 모델 5개에 대해 ~45-60분 (각 모델 ~7-15분)

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# === 진단할 모델 리스트 ===
# 메모리 순서대로 (작은 것 → 큰 것). OOM 발생해도 다음 모델 진행.
# A100 80GB에서:
#   360M → 1.7B → 3B → 7B → MoE
DEFAULT_MODELS=(
    "HuggingFaceTB/SmolLM2-360M"
    "HuggingFaceTB/SmolLM2-1.7B"
    "meta-llama/Llama-3.2-3B"
    "Qwen/Qwen2.5-7B"
)
MODELS="${MODELS:-${DEFAULT_MODELS[@]}}"

BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_BATCHES="${NUM_BATCHES:-4}"
MAX_PARAMS="${MAX_PARAMS:-60}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT="${OUTPUT:-results/phase1_modelsweep.json}"

echo "=== Phase 1C: Model Size Trend ==="
echo "Models:"
for m in $MODELS; do
    echo "  - $m"
done
echo "Per-model: bs=$BATCH_SIZE × seq=$SEQ_LEN, $NUM_BATCHES batches, $MAX_PARAMS params"
echo "Output:    $OUTPUT"
echo ""
echo "Note: Llama-3.2-3B requires HF login. If you haven't:"
echo "  huggingface-cli login"
echo ""

python scripts/phase1_modelsweep.py \
    --models $MODELS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --num_batches $NUM_BATCHES \
    --max_params $MAX_PARAMS \
    --ks 8 32 \
    --alphas 0.5 0.25 0.125 \
    --cert_threshold 0.05 \
    --dtype bfloat16 \
    --output "$OUTPUT"

echo ""
echo "=== Done. Review TREND ANALYSIS above. ==="
