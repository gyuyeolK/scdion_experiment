#!/usr/bin/env bash
# Phase 2: 옵티마이저 비교 학습 - A100 40GB × 4 (PCIe) 환경.
#
# PCIe 환경의 특징:
# - NVLink 없으므로 FSDP 통신이 NVLink 환경 대비 느림
# - 따라서 Dion2의 통신량 절감(α배)이 NVLink 환경보다 더 큰 wall-clock 이득이 될 수 있음
# - 이 부분이 실제로 측정되는 핵심 지표.
#
# 사용법:
#   bash launch_phase2.sh muon
#   ALPHA=0.5 bash launch_phase2.sh dion2_uniform
#   ALPHA=0.5 bash launch_phase2.sh sc_dion

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

OPTIMIZER="${1:-muon}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"

# 학습 길이 (4 GPU + bs=2 + seq=2048 = 16k tokens/step)
# 2000 step ≈ 32M tokens (충분히 길어야 의미 있는 loss 도달, 짧은 학습이지만 비교에는 OK)
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

# Per-GPU batch. 1.7B model이면 bf16에서 안전
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

# 옵티마이저 옵션
ALPHA="${ALPHA:-0.5}"
ALPHA_D="${ALPHA_D:-1.0}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
CERT_THRESHOLD="${CERT_THRESHOLD:-0.05}"
REFRESH_PERIOD="${REFRESH_PERIOD:-10}"

# Learning rates
LR="${LR:-2e-3}"
LR_ADAMW="${LR_ADAMW:-3e-4}"
LR_MIN="${LR_MIN:-2e-4}"

# Output
TAG="${TAG:-${OPTIMIZER}_a${ALPHA}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/$TAG}"

# 환경 변수
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# NCCL: PCIe 환경에서 종종 성능 향상
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"   # PCIe P2P 사용 시도
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"     # InfiniBand 없음

# GPU 자동 감지
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "=== Phase 2 Training ==="
echo "Optimizer:    $OPTIMIZER (α=$ALPHA)"
echo "Model:        $MODEL"
echo "GPUs:         $NGPU (FSDP)"
echo "Max steps:    $MAX_STEPS"
echo "Batch:        $BATCH_SIZE/GPU × seq $SEQ_LEN × accum $GRAD_ACCUM"
echo "             = effective batch $(($BATCH_SIZE * $NGPU * $GRAD_ACCUM)) × $SEQ_LEN tokens"
echo "Output:       $OUTPUT_DIR"
echo

torchrun --standalone --nproc_per_node=$NGPU \
    scripts/phase2_train.py \
    --model_name "$MODEL" \
    --optimizer "$OPTIMIZER" \
    --alpha $ALPHA \
    --alpha_d $ALPHA_D \
    --subspace_rank $SUBSPACE_RANK \
    --cert_threshold $CERT_THRESHOLD \
    --refresh_period $REFRESH_PERIOD \
    --lr $LR \
    --lr_min $LR_MIN \
    --lr_adamw $LR_ADAMW \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --grad_accum_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --eval_interval $EVAL_INTERVAL \
    --dtype bfloat16 \
    --use_fsdp \
    --output_dir "$OUTPUT_DIR"

echo
echo "Done. Results in $OUTPUT_DIR/"
echo ""
echo "여러 run 비교:"
echo "  python analysis/analyze_results.py runs/*/"
