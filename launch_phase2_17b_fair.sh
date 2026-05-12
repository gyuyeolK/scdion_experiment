#!/usr/bin/env bash
# Fair eval 학습 비교: 모든 옵티마이저 같은 eval batches로 평가
#
# 사용: bash launch_phase2_17b_fair.sh

cd "$(dirname "${BASH_SOURCE[0]}" )"

# 비교할 설정: Muon, Dion2(α=0.5), SC-Dion(α=0.5,0.25,0.125)
CONFIGS=(
    "muon|0.5"
    "dion2_uniform|0.5"
    "sc_dion|0.5"
    "sc_dion|0.25"
    "sc_dion|0.125"
)

MAX_STEPS="${MAX_STEPS:-200}"
WARMUP="${WARMUP:-20}"
SEED="${SEED:-42}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_N_BATCHES="${EVAL_N_BATCHES:-4}"

echo "=== Fair Eval Comparison ==="
echo "Configs: ${CONFIGS[@]}"
echo "Steps: $MAX_STEPS, Eval every $EVAL_INTERVAL steps on $EVAL_N_BATCHES fixed batches"
echo ""

t_start=$(date +%s)
n_done=0
n_skip=0
n_fail=0

for cfg in "${CONFIGS[@]}"; do
    OPT="${cfg%|*}"
    ALPHA="${cfg#*|}"
    TAG="${OPT}_a${ALPHA}_fair"
    OUTPUT_DIR="runs_17b_fair/$TAG"
    
    if [ -f "$OUTPUT_DIR/history.json" ]; then
        echo ">>> [SKIP] $TAG"
        n_skip=$((n_skip + 1))
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo ">>> [$(date +%H:%M:%S)] Running: $OPT, α=$ALPHA"
    echo "============================================================"
    
    python scripts/phase2_fair_eval.py \
        --optimizer "$OPT" \
        --alpha $ALPHA \
        --max_steps $MAX_STEPS \
        --warmup_steps $WARMUP \
        --seed $SEED \
        --eval_interval $EVAL_INTERVAL \
        --eval_n_batches $EVAL_N_BATCHES \
        --output_dir "$OUTPUT_DIR" \
        --log_interval 25 \
        || true
    
    if [ -f "$OUTPUT_DIR/history.json" ]; then
        n_done=$((n_done + 1))
        echo ">>> [OK] $TAG"
    else
        n_fail=$((n_fail + 1))
        echo ">>> [FAIL] $TAG"
    fi
done

t_end=$(date +%s)
elapsed=$(((t_end - t_start) / 60))

echo ""
echo "============================================================"
echo "=== Fair eval done in $elapsed min ==="
echo "    Completed: $n_done, Skipped: $n_skip, Failed: $n_fail"
echo "============================================================"

echo ""
echo "Running fair-eval analysis..."
python analysis/fair_eval_analysis.py runs_17b_fair/ || true
