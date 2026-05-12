#!/usr/bin/env bash
# Phase 1: 학습 전 진단 - A100 40GB × 4 (PCIe) 환경 최적화.
#
# 사용법:
#   bash launch_phase1.sh                                       # 기본: 1.7B
#   MODEL=HuggingFaceTB/SmolLM2-360M bash launch_phase1.sh       # 더 작게
#   MODEL=meta-llama/Llama-3.2-3B bash launch_phase1.sh          # 3B
#   MODEL=Qwen/Qwen2.5-7B bash launch_phase1.sh                  # 7B
#   MODEL=Qwen/Qwen1.5-MoE-A2.7B bash launch_phase1.sh           # MoE 시도

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# === 모델 선택 ===
# A100 40GB 단일 GPU 메모리 예상 (bf16 + gradient checkpointing + bs=2 seq=2048):
#   SmolLM2-360M:  ~5 GB  ✓ 매우 여유
#   SmolLM2-1.7B:  ~14 GB ✓ 여유 (권장 시작점)
#   Llama-3.2-3B:  ~22 GB ✓ 가능
#   Qwen2.5-7B:    ~38 GB ⚠ 빠듯 (배치 키우지 말 것)
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"

# A100 40GB이면 처음 설정으로 무리 없음
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_BATCHES="${NUM_BATCHES:-4}"
USE_FINEWEB="${USE_FINEWEB:-1}"

# 진단은 single-GPU로 충분 (GPU 0번 사용)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 출력 파일명에 모델명 포함 (여러 모델 비교 시 편함)
MODEL_TAG=$(basename "$MODEL")
OUTPUT="${OUTPUT:-results/phase1_${MODEL_TAG}.json}"

echo "=== Phase 1 Diagnostic ==="
echo "Model:        $MODEL"
echo "GPU:          CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Batch:        $BATCH_SIZE × seq $SEQ_LEN, $NUM_BATCHES batches"
echo "Output:       $OUTPUT"
echo

FW_FLAG=""
if [ "$USE_FINEWEB" = "1" ]; then
    FW_FLAG="--use_fineweb"
fi

python scripts/phase1_diagnose.py \
    --model_name "$MODEL" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --num_batches $NUM_BATCHES \
    --ks 4 8 16 32 64 \
    --alphas 0.5 0.25 0.125 \
    --cert_threshold 0.05 \
    --dtype bfloat16 \
    $FW_FLAG \
    --output "$OUTPUT"

echo
echo "=== 다음 단계 가이드 ==="
echo ""
echo "결과 보기:"
echo "  cat $OUTPUT | python -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d[\"summary\"], indent=2))'"
echo ""
echo "Phase 1을 더 큰 모델로 반복:"
echo "  MODEL=meta-llama/Llama-3.2-3B   bash launch_phase1.sh"
echo "  MODEL=Qwen/Qwen2.5-7B           bash launch_phase1.sh"
echo "  MODEL=Qwen/Qwen1.5-MoE-A2.7B    bash launch_phase1.sh   # MoE는 다를 가능성"
echo ""
echo "Phase 1 결과가 유망하면 Phase 2 학습 비교:"
echo "  bash launch_phase2.sh muon"
echo "  bash launch_phase2.sh dion2_uniform"
echo "  bash launch_phase2.sh sc_dion"
