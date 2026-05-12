#!/usr/bin/env bash
# Phase 1C: Model size sweep.
# 여러 모델 크기를 자동 진단해서 SC-Dion trend 분석.
#
# 80GB GPU 메모리 예상:
#   SmolLM2-1.7B:   ~14 GB (이미 진단됨, 비교용)
#   Llama-3.2-3B:   ~22 GB
#   Qwen2.5-7B:     ~38 GB  (메모리 OK)
#   Llama-3.1-8B:   ~42 GB  (메모리 OK)
#
# 사용:
#   bash launch_phase1_sweep.sh                    # 기본 set
#   MODELS="HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B" \
#       bash launch_phase1_sweep.sh                # 커스텀 set

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 기본 모델 set: 모두 공개 모델 (인증 불필요).
# 1.7B는 이미 진단됐으므로 trend 분석을 위한 비교군:
#   - SmolLM2-135M, 360M, 1.7B: 같은 family, 크기만 다름 → 순수한 size effect
#   - Qwen2.5-3B, 7B: 다른 family, 더 큼
# 인증 필요한 모델 (Llama 등)을 쓰려면: huggingface-cli login 먼저
DEFAULT_MODELS="HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B"
MODELS="${MODELS:-$DEFAULT_MODELS}"

# 빠른 진단용 설정
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"      # 더 짧게 (메모리 절약)
NUM_BATCHES="${NUM_BATCHES:-4}"
MAX_PARAMS="${MAX_PARAMS:-80}"  # GPU 진단은 빠르니까 더 많이

KS="${KS:-8 32}"
ALPHAS="${ALPHAS:-0.5 0.25 0.125}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

RESULTS_DIR="${RESULTS_DIR:-results/sweep}"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 1C: Model Size Sweep ==="
echo "Models: $MODELS"
echo "Results: $RESULTS_DIR"
echo ""

for MODEL in $MODELS; do
    MODEL_TAG=$(basename "$MODEL")
    OUTPUT="$RESULTS_DIR/${MODEL_TAG}.json"
    
    if [ -f "$OUTPUT" ]; then
        echo "[$MODEL_TAG] Already exists, skipping. Delete $OUTPUT to redo."
        continue
    fi
    
    echo ""
    echo "========================================================================"
    echo "Diagnosing: $MODEL"
    echo "========================================================================"
    
    # 모델별 메모리 안전장치: 너무 크면 seq 더 줄임
    EFFECTIVE_SEQ=$SEQ_LEN
    if [[ "$MODEL" == *"7B"* ]] || [[ "$MODEL" == *"8B"* ]]; then
        EFFECTIVE_SEQ=512
        echo "(Large model detected, reducing seq_len to $EFFECTIVE_SEQ)"
    fi
    
    # || true: 한 모델 실패해도 나머지 진행
    python scripts/phase1_sweep.py \
        --model_name "$MODEL" \
        --batch_size $BATCH_SIZE \
        --seq_len $EFFECTIVE_SEQ \
        --num_batches $NUM_BATCHES \
        --ks $KS \
        --alphas $ALPHAS \
        --cert_threshold 0.05 \
        --dtype bfloat16 \
        --max_params $MAX_PARAMS \
        --output "$OUTPUT" || {
        echo "[$MODEL_TAG] FAILED. Continuing to next model."
        continue
    }
    
    echo ""
    echo "[$MODEL_TAG] DONE → $OUTPUT"
    
    # GPU 메모리 해제 위해 잠깐 sleep
    sleep 5
done

echo ""
echo "========================================================================"
echo "All done. Comparing across models:"
echo "========================================================================"

python scripts/compare_sweep.py "$RESULTS_DIR"
