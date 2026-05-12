# SC-Dion 실증 결과 한 페이지 요약

## TL;DR

**SmolLM2-1.7B continued pretraining에서 SC-Dion이 Muon 대비 wall-clock 40.9% 절감** (α=0.5), 품질 손실 0, 4 seeds에서 σ=0.1% 재현성. α=0.125까지 줄이면 53.6% 절감 (이론 floor 54.2% 근처).

## 데이터

### 1. 메인 비교 (1.7B, seed 42, 300 steps)

```
Optimizer            Step time   Ratio    Total time   Final eval
─────────────────────────────────────────────────────────────────
Muon                 870 ms      1.000x   265 s        2.3902
Dion2-uniform α=0.5  567 ms      0.652x   174 s        2.3907
SC-Dion α=0.5        515 ms      0.591x   165 s        2.3899
```

### 2. Seed robustness (4 seeds)

```
Optimizer            Mean ratio   σ      Pass rate
─────────────────────────────────────────────────
Muon                 1.000        ±0.1%  -
Dion2-uniform α=0.5  0.652        ±0.1%  -
SC-Dion α=0.5        0.591        ±0.1%  100%
```

세 옵티마이저 모두 4 seeds에서 동일한 loss 도달 (Δ < 0.0005).

### 3. α scaling (seed 42)

```
α       Step time   Ratio    Savings   Pass rate   Predicted   Error
────────────────────────────────────────────────────────────────────
0.5     515 ms      0.591    40.9%     100%        0.593       +0.2%
0.25    431 ms      0.495    50.5%     100%        0.492       -0.7%
0.125   404 ms      0.464    53.6%     99.9%       0.466       +0.4%
```

이론 모델: ratio(α) = 0.458 + 0.542·α² (1% 이내 fit)

### 4. Fair eval (fixed eval batches)

```
Config              Step ms   Final eval   Δ vs Muon
────────────────────────────────────────────────────
Muon                943       2.3902       (baseline)
Dion2 α=0.5         640       2.3907       +0.0004
SC-Dion α=0.5       588       2.3899       -0.0004
SC-Dion α=0.25      504       2.3900       -0.0003
SC-Dion α=0.125     477       2.3903       +0.0000
```

모든 옵티마이저가 동일한 eval batches에서 평가됨. **Eval loss Δ < 0.0006** = 측정 noise. **Free lunch 확인**.

### 5. Phase 1 진단

학습 중 1.7B, 1000 steps, 6 시점 측정:
- Step 0~1000 모든 시점에서 통과율 100%
- τ ≈ 0.19, ω ≈ 0.60 (모멘텀 누적 S_t)
- G_t (단순 그래디언트)도 통과 100%, 하지만 ω 약간 낮음

## ρ_opt 검증

논문의 핵심 진단 변수 ρ_opt = optimizer_time / total_step_time:

| Model | ρ_opt | α=0.5 wall-clock 이득 |
|---|---|---|
| 360M | ~0.10 | 0% (no win) |
| **1.7B** | **~0.45** | **40.9%** |

**360M에서 win 없음은 논문이 정확히 예측한 시나리오**. ρ_opt가 작으면 옵티마이저 절감이 wall-clock에 안 보임. 1.7B에서 ρ_opt가 충분히 큼 → win 발현.

## 솔직한 의미

본 실증이 답한 것:
- "이 논문 합성 환경 외에서도 작동하는가?" → **Yes (1.7B에서)**
- "Quality 손실 없는가?" → **Yes (eval Δ < 0.001)**
- "이론 예측이 정확한가?" → **Yes (1% 이내)**

본 실증이 답하지 못한 것:
- From-scratch 사전학습에서도 같은가?
- 분산 학습 (FSDP) 통신 절감은 얼마나 큰가?
- 10k+ steps 장기 학습 안정성?
- 다른 architecture (MoE 등) 작동?

특히 분산 학습은 논문의 강조점이지만 single GPU로는 검증 불가. 이건 정직히 future work.

## 재현 명령 (최소)

```bash
# 가장 빠른 1.7B 비교 (~15분)
bash launch_phase2_17b.sh muon
bash launch_phase2_17b.sh sc_dion
python analysis/analyze_phase2_small.py runs_17b/muon_a0.5_topk runs_17b/sc_dion_a0.5_topk
```

전체 실험 (~3시간)은 README.md 참조.
