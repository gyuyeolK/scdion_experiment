#!/usr/bin/env bash
# Alpha scaling 검증: SC-Dion at α=0.5, 0.25, 0.125
#
# 목적: α 줄일수록 wall-clock 더 빨라지는지, 이론 (α²) 따르는지 검증.
#
# Muon baseline은 이미 있음 (runs_17b/muon_a0.5_topk).
# SC-Dion α=0.5도 있음 (runs_17b/sc_dion_a0.5_topk).
# 추가로 α=0.25, 0.125만 학습.
#
# 시간: 2 run × ~3분 = ~6분

cd "$(dirname "${BASH_SOURCE[0]}" )"

ALPHAS="${ALPHAS:-0.25 0.125}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-200}"

echo "=== Alpha Scaling Experiment ==="
echo "SC-Dion at α: $ALPHAS"
echo "Seed: $SEED, Steps: $MAX_STEPS"
echo ""

for ALPHA in $ALPHAS; do
    TAG="sc_dion_a${ALPHA}_topk"
    OUTPUT_DIR="runs_17b_alpha/$TAG"
    
    if [ -f "$OUTPUT_DIR/history.json" ]; then
        echo ">>> [SKIP] $TAG (already done)"
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo ">>> [$(date +%H:%M:%S)] Running: sc_dion, α=$ALPHA"
    echo "============================================================"
    
    ALPHA=$ALPHA \
    SEED=$SEED \
    MAX_STEPS=$MAX_STEPS \
    WARMUP=20 \
    TAG=$TAG \
    OUTPUT_DIR=$OUTPUT_DIR \
        bash launch_phase2_17b.sh sc_dion || true
    
    if [ -f "$OUTPUT_DIR/history.json" ]; then
        echo ">>> [OK] $TAG"
    else
        echo ">>> [FAIL] $TAG"
    fi
done

# Copy baselines for comparison (linked to one place)
mkdir -p runs_17b_alpha
[ ! -e runs_17b_alpha/muon ] && [ -d runs_17b/muon_a0.5_topk ] && \
    ln -s ../runs_17b/muon_a0.5_topk runs_17b_alpha/muon 2>/dev/null
[ ! -e runs_17b_alpha/sc_dion_a0.5_topk ] && [ -d runs_17b/sc_dion_a0.5_topk ] && \
    ln -s ../runs_17b/sc_dion_a0.5_topk runs_17b_alpha/sc_dion_a0.5_topk 2>/dev/null
[ ! -e runs_17b_alpha/dion2_uniform_a0.5_topk ] && [ -d runs_17b/dion2_uniform_a0.5_topk ] && \
    ln -s ../runs_17b/dion2_uniform_a0.5_topk runs_17b_alpha/dion2_uniform_a0.5_topk 2>/dev/null

echo ""
echo "=== Alpha sweep done ==="
echo ""
echo "Running alpha scaling analysis..."
python analysis/alpha_scaling.py runs_17b_alpha/
