#!/usr/bin/env bash
# 시드 robustness 검증 - robust 버전.
# set -e 제거 (exit code 비정상이라도 다음 run 진행).
# 각 run의 결과는 history.json에 저장되므로 중간에 끊겨도 손실 없음.
#
# 사용: bash launch_phase2_17b_seedsweep.sh
#
# 이미 끝난 run은 자동 skip → 중간 끊김 후 재실행 OK.

# NOTE: set -e 의도적으로 제외 — 한 run이 비정상 exit code 반환해도 다음 진행
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-1 7 123}"
OPTIMIZERS="${OPTIMIZERS:-muon dion2_uniform sc_dion}"
MAX_STEPS="${MAX_STEPS:-200}"
WARMUP="${WARMUP:-20}"

echo "=== Seed Robustness Sweep (robust mode) ==="
echo "Seeds:       $SEEDS"
echo "Optimizers:  $OPTIMIZERS"
echo "Steps each:  $MAX_STEPS"
echo ""

t_start=$(date +%s)
n_done=0
n_skip=0
n_fail=0

for SEED in $SEEDS; do
    for OPT in $OPTIMIZERS; do
        TAG="${OPT}_s${SEED}"
        OUTPUT_DIR="runs_17b_seeds/$TAG"
        
        if [ -f "$OUTPUT_DIR/history.json" ]; then
            echo ">>> [SKIP] $TAG (already done)"
            n_skip=$((n_skip + 1))
            continue
        fi
        
        echo ""
        echo "============================================================"
        echo ">>> [$(date +%H:%M:%S)] Running: $OPT, seed=$SEED"
        echo "============================================================"
        
        # || true로 exit code 무시
        SEED=$SEED \
        MAX_STEPS=$MAX_STEPS \
        WARMUP=$WARMUP \
        TAG=$TAG \
        OUTPUT_DIR=$OUTPUT_DIR \
            bash launch_phase2_17b.sh $OPT || true
        
        # 결과 파일이 생겼는지로 성공 판단
        if [ -f "$OUTPUT_DIR/history.json" ]; then
            n_done=$((n_done + 1))
            echo ">>> [OK] $TAG completed"
        else
            n_fail=$((n_fail + 1))
            echo ">>> [FAIL] $TAG did not produce history.json"
        fi
    done
done

t_end=$(date +%s)
elapsed=$(((t_end - t_start) / 60))

echo ""
echo "============================================================"
echo "=== Sweep finished in $elapsed min ==="
echo "    Completed: $n_done, Skipped: $n_skip, Failed: $n_fail"
echo "============================================================"
echo ""

# 결과 디렉토리 요약
echo "Result directories:"
ls runs_17b_seeds/ 2>/dev/null || echo "  (no results yet)"
echo ""

# 분석 실행
echo "Running seed_analysis..."
python analysis/seed_analysis.py runs_17b_seeds/ || true
