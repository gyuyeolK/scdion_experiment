# SC-Dion Wall-Clock Validation (1.7B Continued Pretraining)

논문 "Certified Selector-aware Dion" (SC-Dion)의 wall-clock 주장을 실제 LLM (SmolLM2-1.7B)에서 검증한 실험. 합성 환경 외의 실제 학습 환경에서 wall-clock 이득을 처음 실증.

## 핵심 결과

| Optimizer | Step time | Wall-clock vs Muon | Eval loss Δ | Verdict |
|---|---|---|---|---|
| Muon (baseline) | 870 ms | 1.000× | 0 | baseline |
| Dion2-uniform α=0.5 | 567 ms | **0.652×** (34.8% 빠름) | +0.0002 | 🏆 CLEAN WIN |
| SC-Dion α=0.5 | 515 ms | **0.591×** (40.9% 빠름) | +0.0001 | 🏆 CLEAN WIN |
| SC-Dion α=0.25 | 431 ms | **0.495×** (50.5% 빠름) | -0.0003 | 🏆 CLEAN WIN |
| SC-Dion α=0.125 | 404 ms | **0.464×** (53.6% 빠름) | +0.0000 | 🏆 CLEAN WIN |

측정 환경: SmolLM2-1.7B, FineWeb 데이터, batch=2, seq=2048, NVIDIA A100 80GB, bf16, 200-300 steps.

### Reproducibility (4 seeds: 1, 7, 42, 123)

| Optimizer | Wall-clock ratio | σ |
|---|---|---|
| Dion2-uniform α=0.5 | 0.652 ± 0.001 | CV=0.1% ⭐ |
| SC-Dion α=0.5 | 0.591 ± 0.001 | CV=0.1% ⭐ |

CV=0.1%는 사실상 결정론적인 재현성.

### 이론 검증 (α² scaling)

논문 wall-clock identity: ratio(α) = ρ_fb + α²·ρ_flop + α·ρ_byte

Single GPU에서 ρ_byte ≈ 0으로 fitting:
- ρ_fb = 0.458 (forward+backward 비중)
- ρ_flop = 0.542 (optimizer FLOP 비중)
- 최대 가능 절감 (α→0): 54.2%
- α=0.125에서 53.6% 절감 (floor 거의 도달)

예측 vs 실측: 1% 이내 일치

| α | Predicted | Actual | Error |
|---|---|---|---|
| 0.5 | 0.593 | 0.592 | +0.2% |
| 0.25 | 0.492 | 0.495 | -0.7% |
| 0.125 | 0.466 | 0.464 | +0.4% |

## 실험 단계

### Phase 1: 그래디언트 구조 진단

증명서 ĉ = √ω(1-τ) - τ가 실제 모델에서 통과하는지 확인.

Phase 1A: 모델별 진단 (t=0)
- SmolLM2-360M, SmolLM2-1.7B, Qwen2.5-7B
- 결과: 모든 (k, α) 조합에서 통과율 100%

Phase 1B: 데이터 소스 robustness
- random, FineWeb, Wikipedia
- 결과: 데이터 소스 무관 (차이 < 3.3%)

Phase 1C: 학습 중 진단 (1.7B, 1000 steps)
- 6개 시점 (step 0, 10, 50, 200, 500, 1000)에서 측정
- 결과: 통과율 100% 유지, τ≈0.19, ω≈0.60 안정적
- G_t vs S_t (모멘텀 누적): S_t가 약간 더 좋음 (논문 가정 일치)

### Phase 2: Wall-clock 비교

360M (실패한 케이스):
- ρ_opt가 작아서 wall-clock 차이 미미
- Muon 342ms vs SC-Dion 354ms (1.035×)
- 논문의 "ρ_opt 작으면 win 없음" 진단 정확히 재현

1.7B (성공):
- ρ_opt 큼 → α=0.5에서 40.9% 절감
- 4 seeds (1, 7, 42, 123)에서 σ=0.1%로 reproducible

### Phase 3: 추가 검증

α scaling: 0.5 → 0.25 → 0.125에서 monotonic 빨라짐, α² 이론 일치  
Fair eval: 고정 eval batches로 비교, eval loss Δ < 0.0006 (free lunch 확인)

## 환경

- GPU: NVIDIA A100-SXM4-80GB (single GPU 사용)
- PyTorch: 2.11.0+cu128
- Python: 3.12
- 모델: HuggingFaceTB/SmolLM2-{360M,1.7B}, Qwen2.5-7B
- 데이터: HuggingFaceFW/fineweb (sample-10BT, streaming)
- dtype: bfloat16

## 재현하는 법

### 환경 준비

```bash
# vast.ai A100 80GB instance 권장
pip install torch torchvision transformers datasets

# HF cache를 큰 디스크로 (overlay 작은 환경에서)
export HF_HOME=/dev/shm/hf_cache
export HF_HUB_CACHE=/dev/shm/hf_cache
```

### 학습 비교 (가장 빠른 핵심 결과, ~15분)

```bash
cd scdion_experiment

# 1.7B, 3 옵티마이저, 300 step씩
bash launch_phase2_17b.sh muon
bash launch_phase2_17b.sh dion2_uniform
bash launch_phase2_17b.sh sc_dion

# 분석
python analysis/analyze_phase2_small.py runs_17b/*_topk
```

### Seed robustness (4 seeds, ~30분)

```bash
bash launch_phase2_17b_seedsweep.sh
# 자동으로 seed_analysis.py 실행
```

### α scaling (3 α 값, ~10분)

```bash
bash launch_phase2_17b_alphasweep.sh
# 자동으로 alpha_scaling.py 실행
```

### Fair eval (5 configs, ~15분)

```bash
bash launch_phase2_17b_fair.sh
# 자동으로 fair_eval_analysis.py 실행
```

### Phase 1 진단 (선택, ~1.5시간)

```bash
# 학습 중 진단 (가장 중요)
bash launch_phase1_intraining.sh

# 다양한 모델
bash launch_phase1_modelsweep.sh
```

## 코드 구조

```
scdion_experiment/
├── optimizers/
│   ├── muon.py              # Standard Muon
│   ├── dion2.py             # Dion2-uniform (random row selection)
│   ├── sc_dion.py           # SC-Dion (CPU reference)
│   ├── sc_dion_gpu.py       # SC-Dion GPU (production)
│   └── newton_schulz.py     # NS orthogonalization (bf16 native)
├── scripts/
│   ├── phase1_diagnose.py        # t=0 진단
│   ├── phase1_modelsweep.py      # 다중 모델
│   ├── phase1_intraining.py      # 학습 중 진단
│   ├── phase2_small.py           # 학습 비교
│   ├── phase2_fair_eval.py       # 공정 평가
│   └── validate_sc_dion_gpu.py   # GPU 구현 검증
├── analysis/
│   ├── analyze_phase2_small.py   # 학습 결과 분석
│   ├── seed_analysis.py          # 시드 robustness
│   ├── alpha_scaling.py          # α 이론 검증
│   └── fair_eval_analysis.py     # Fair eval 분석
└── launch_*.sh                   # 실행 스크립트
```

## SC-Dion GPU 구현 핵심

3가지 selector 옵션 (`--selector`):
- `topk` (기본): row L2 norm top-k. 가장 빠름. 학습에 권장.
- `block_greedy`: Woodbury block update. 정확도 중간.
- `greedy`: Sherman-Morrison incremental. 정확도 최고, 느림.

핵심 최적화:
- bf16 native NS (A100 tensor core 활용)
- Subspace + selection을 K=20 steps마다 cache
- `_randomized_subspace_gpu` GPU 완전 vectorized

## 한계 & 향후 과제

본 검증의 범위:
- Continued pretraining (사전학습된 모델에서 추가 학습)
- Single GPU
- 200-300 step (수렴 trend만 측정)
- SmolLM2 architecture

미검증:
- From-scratch 사전학습 (모델을 처음부터 학습)
- 분산 학습 (FSDP 등에서의 통신 절감)
- 장기 학습 (10k+ step에서의 안정성)
- 다른 architecture (MoE, Mamba 등)
- 다양한 task에서의 downstream 평가

특히 분산 학습은 논문이 강조한 시나리오인데 single GPU로는 검증 불가.

## 결과 파일 위치

모든 결과는 JSON으로 저장됨:

| 디렉토리 | 내용 |
|---|---|
| `runs_17b/*/history.json` | 1.7B 기본 학습 (seed 42) |
| `runs_17b_seeds/*/history.json` | 4 seeds robustness |
| `runs_17b_alpha/*/history.json` | α scaling 실험 |
| `runs_17b_fair/*/history.json` | Fair eval (fixed eval batches) |
| `results/phase1_*.json` | Phase 1 진단 결과 |

각 history.json은 step_times, loss curve, SC-Dion 통계, config 포함.

## 솔직한 평가

본 실증의 의미:
- 논문의 핵심 wall-clock 주장이 실제 1.7B 학습에서 작동함을 처음 실증
- 합성 환경 외에서 wall-clock 41-54% 절감, quality 손실 0
- 이론 (Cor 4.3) 1% 이내 정확한 fit
- 4 seeds × 3 옵티마이저로 reproducibility 확립

본 실증의 caveat:
- Continued pretraining 한정 (from-scratch 미검증)
- Single GPU (통신 절감 측정 안 됨)
- Short horizon (200-300 step)
- Loss는 EMA 아닌 fixed eval batch로 측정 (artifact 제거)
