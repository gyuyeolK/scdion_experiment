"""
Compare diagnostic results across model sizes. 

trend 분석:
- 모델 크기 → pass rate, τ, ω, row_norm_spread, stable_rank 변화
- "SC-Dion이 큰 모델에서 더 잘 작동하는가?"의 답

사용:
    python scripts/compare_sweep.py results/sweep/
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob('*.json'))
    
    if not files:
        print(f"No JSON results found in {results_dir}")
        return
    
    models = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            models.append(data)
        except Exception as e:
            print(f"Skip {f}: {e}")
    
    if not models:
        print("No valid result files")
        return
    
    # Sort by total_params
    models.sort(key=lambda m: m.get('total_params_B', 0))
    
    print("=" * 100)
    print("MODEL SIZE SWEEP COMPARISON")
    print("=" * 100)
    
    # Basic info table
    print(f"\n{'Model':<35} {'Size':<8} {'Loss':<8} {'Spread':<10} {'SR med':<10} "
          f"{'Diag time':<10}")
    print("-" * 100)
    for m in models:
        name = m['model_name'][:33]
        size = f"{m['total_params_B']:.2f}B"
        loss = m['mean_loss']
        spread = m.get('row_norm_spread', {}).get('median', 0)
        sr = m.get('stable_rank', {}).get('median', 0)
        diag = m.get('diagnostic_time_sec', 0)
        print(f"{name:<35} {size:<8} {loss:<8.3f} {spread:<10.1f} {sr:<10.2f} "
              f"{diag:<10.0f}")
    
    # Get all (k, α) configs
    all_configs = set()
    for m in models:
        all_configs.update(m.get('per_config', {}).keys())
    configs = sorted(all_configs)
    
    # For each config, show trend
    print(f"\n{'='*100}")
    print("Pass rate trend by model size (key metric for SC-Dion viability)")
    print(f"{'='*100}")
    
    for config in configs:
        print(f"\n--- {config} ---")
        print(f"{'Model':<35} {'Size':<8} {'Pass %':<10} {'τ':<8} {'ω':<8} {'cert':<10}")
        print("-" * 90)
        for m in models:
            pc = m.get('per_config', {}).get(config)
            if pc is None:
                continue
            name = m['model_name'][:33]
            size = f"{m['total_params_B']:.2f}B"
            print(f"{name:<35} {size:<8} {100*pc['pass_rate']:<10.1f} "
                  f"{pc['mean_tau']:<8.3f} {pc['mean_omega']:<8.3f} "
                  f"{pc['mean_cert']:+.3f}")
    
    # Trend analysis
    print(f"\n{'='*100}")
    print("TREND ANALYSIS")
    print(f"{'='*100}")
    
    if len(models) < 2:
        print("Need at least 2 models for trend analysis.")
        return
    
    # 가장 적극적인 설정에서 trend 추적
    aggressive_config = None
    for cand in ['k8_a0.125', 'k32_a0.125', 'k8_a0.25', 'k32_a0.25']:
        if cand in configs:
            aggressive_config = cand
            break
    
    if aggressive_config:
        print(f"\nMost aggressive config: {aggressive_config}")
        sizes_pass = []
        for m in models:
            pc = m.get('per_config', {}).get(aggressive_config)
            if pc:
                sizes_pass.append((m['total_params_B'], pc['pass_rate'],
                                   pc['mean_tau'], pc['mean_omega']))
        if len(sizes_pass) >= 2:
            print(f"\n  Size growth vs pass rate:")
            for size, pr, tau, om in sizes_pass:
                bar = "█" * int(pr * 30)
                print(f"    {size:.2f}B  {100*pr:>5.1f}%  τ={tau:.3f}  ω={om:.3f}  {bar}")
            
            # Trend direction
            if sizes_pass[-1][1] > sizes_pass[0][1] + 0.1:
                print(f"\n  📈 GROWING: Pass rate increases with model size")
                print(f"     → 큰 LLM일수록 SC-Dion에 유리한 그래디언트 구조")
            elif sizes_pass[-1][1] < sizes_pass[0][1] - 0.1:
                print(f"\n  📉 SHRINKING: Pass rate decreases with model size")
                print(f"     → 큰 모델은 grad가 더 fuller-rank")
            else:
                print(f"\n  ➡️ FLAT: Pass rate roughly constant across sizes")
                print(f"     → 모델 크기와 무관하게 SC-Dion 가설 작동")
    
    # tau trend
    print(f"\nDominant-tail ratio τ trend (낮을수록 더 저랭크, SC-Dion에 유리):")
    for config in [c for c in configs if c.startswith('k32')]:
        print(f"  {config}:")
        for m in models:
            pc = m.get('per_config', {}).get(config)
            if pc:
                print(f"    {m['total_params_B']:>5.2f}B: τ={pc['mean_tau']:.3f}")
    
    # Stable rank trend
    print(f"\nStable rank trend (낮을수록 더 저랭크):")
    for m in models:
        sr = m.get('stable_rank', {}).get('median', 0)
        print(f"  {m['total_params_B']:>5.2f}B: SR_median={sr:.2f}")
    
    # Final recommendation
    print(f"\n{'='*100}")
    print("RECOMMENDATION")
    print(f"{'='*100}")
    
    # Best (most aggressive but still passing) config across all models
    best_models = {}  # (size, model_name) -> (config, pass_rate)
    for m in models:
        size = m['total_params_B']
        per_config = m.get('per_config', {})
        # Find most aggressive (smallest alpha) that still has >70% pass
        for alpha in [0.125, 0.25, 0.5]:
            for k in [32, 8]:
                cand = f"k{k}_a{alpha}"
                pc = per_config.get(cand)
                if pc and pc['pass_rate'] >= 0.7:
                    best_models[(size, m['model_name'])] = (cand, pc['pass_rate'])
                    break
            if (size, m['model_name']) in best_models:
                break
    
    if best_models:
        print("\nBest aggressive config per model (pass rate ≥ 70%):")
        for (size, name), (config, pr) in sorted(best_models.items()):
            print(f"  {name[:40]:<40} {size:.2f}B  →  {config}  ({100*pr:.0f}%)")
    
    # Largest model with high pass rate
    largest_strong = None
    for m in sorted(models, key=lambda x: -x['total_params_B']):
        for alpha in [0.125, 0.25]:
            pc = m.get('per_config', {}).get(f"k8_a{alpha}")
            if pc and pc['pass_rate'] >= 0.9:
                largest_strong = (m['model_name'], m['total_params_B'], alpha,
                                 pc['pass_rate'])
                break
        if largest_strong:
            break
    
    if largest_strong:
        name, size, alpha, pr = largest_strong
        print(f"\n🎯 Phase 2 추천 모델: {name}")
        print(f"   ({size:.1f}B, α={alpha}에서 {100*pr:.0f}% 통과)")
        print(f"   → 이 모델 + α={alpha}로 학습 비교 진행 권장")
    else:
        print("\n⚠️ 어떤 모델도 α=0.125-0.25에서 90% 이상 통과하지 못함.")
        print("   더 큰 모델 (12B+) 시도 또는 알고리즘 자체 재검토 필요.")


if __name__ == '__main__':
    main()
