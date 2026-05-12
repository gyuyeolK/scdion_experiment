"""
여러 Phase 2 run 결과 비교 - 누가 wall-clock에서 이겼는가?

사용:
    python analyze_results.py runs/muon/ runs/dion2_uniform_a0.5/ runs/sc_dion_a0.5/
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_run(path: Path) -> dict:
    """Load history.json from a run directory."""
    hist_file = path / 'history.json'
    if not hist_file.exists():
        return None
    with open(hist_file) as f:
        return json.load(f)


def target_step_and_time(run: dict, target: float) -> tuple:
    """Return (step, elapsed_sec) at first time eval_loss <= target. (None, None) if never."""
    if 'target_hits' in run and str(target) in run['target_hits']:
        info = run['target_hits'][str(target)]
        if info is not None:
            return info['step'], info['elapsed_sec']
    # Fallback: scan history
    for h in run.get('history', []):
        if h.get('eval_loss', float('inf')) <= target:
            return h['step'], h['elapsed_sec']
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', type=str, nargs='+')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Run name to use as baseline (default: first muon run)')
    parser.add_argument('--targets', type=float, nargs='+',
                        default=[5.0, 4.5, 4.0, 3.5, 3.0])
    parser.add_argument('--output', type=str, default='comparison.json')
    args = parser.parse_args()
    
    runs = {}
    for d in args.run_dirs:
        p = Path(d)
        name = p.name
        run = load_run(p)
        if run is None:
            print(f"[skip] {d}: no history.json")
            continue
        runs[name] = run
    
    # Pick baseline
    if args.baseline and args.baseline in runs:
        baseline_name = args.baseline
    else:
        baseline_name = None
        for n in runs:
            if 'muon' in n.lower() and 'dion' not in n.lower():
                baseline_name = n
                break
        if baseline_name is None:
            baseline_name = list(runs.keys())[0]
    print(f"Baseline: {baseline_name}\n")
    
    baseline = runs[baseline_name]
    
    # Per-run summary
    print("=" * 100)
    print(f"{'Run':<40} {'Final loss':>12} {'Total time':>12} {'Avg ms/step':>12} {'ρ_opt':>8}")
    print("=" * 100)
    summary = {}
    for name, run in runs.items():
        hist = run.get('history', [])
        if not hist:
            continue
        final_loss = hist[-1].get('eval_loss', float('nan'))
        total_t = run.get('total_elapsed_sec', hist[-1].get('elapsed_sec', 0))
        avg_step = np.mean([h.get('step_time_ms', 0) for h in hist])
        avg_rho_opt = np.mean([h.get('rho_opt', 0) for h in hist])
        print(f"{name:<40} {final_loss:>12.4f} {total_t/60:>10.1f}m {avg_step:>10.1f}ms {avg_rho_opt:>8.3f}")
        summary[name] = {
            'final_loss': final_loss,
            'total_elapsed_sec': total_t,
            'avg_step_time_ms': float(avg_step),
            'avg_rho_opt': float(avg_rho_opt),
        }
    
    # Target hit comparison: 가장 중요한 표 -- T_D vs T_M, wall-clock 비율
    print("\n" + "=" * 100)
    print("Target loss → time/step comparisons (vs baseline)")
    print("=" * 100)
    
    target_table = {}
    for target in args.targets:
        base_step, base_t = target_step_and_time(baseline, target)
        if base_step is None:
            print(f"\nTarget loss {target}: baseline never reached")
            target_table[target] = {'baseline_reached': False}
            continue
        
        print(f"\nTarget loss {target}:  baseline {baseline_name} hits at step {base_step}, "
              f"time {base_t/60:.1f}m")
        print(f"  {'Optimizer':<40} {'Step':>8} {'T_D/T_M':>10} {'Time':>10} {'WC ratio':>10} "
              f"{'Verdict':>16}")
        target_table[target] = {
            'baseline': {'name': baseline_name, 'step': base_step, 'time_sec': base_t},
            'others': {},
        }
        
        for name, run in runs.items():
            if name == baseline_name:
                continue
            step, t = target_step_and_time(run, target)
            if step is None:
                print(f"  {name:<40} {'-':>8} {'-':>10} {'-':>10} {'-':>10} {'never reached':>16}")
                target_table[target]['others'][name] = {'reached': False}
                continue
            step_ratio = step / base_step
            time_ratio = t / base_t
            verdict = "✅ wall-clock win" if time_ratio < 0.98 else \
                      ("❌ wall-clock loss" if time_ratio > 1.02 else "≈ tie")
            print(f"  {name:<40} {step:>8d} {step_ratio:>10.2f} {t/60:>8.1f}m "
                  f"{time_ratio:>10.3f} {verdict:>16}")
            target_table[target]['others'][name] = {
                'reached': True,
                'step': step,
                'time_sec': t,
                'step_ratio_vs_baseline': step_ratio,
                'time_ratio_vs_baseline': time_ratio,
                'verdict': verdict,
            }
    
    # ρ decomposition - 논문의 wall-clock identity 검증
    print("\n" + "=" * 100)
    print("Wall-clock identity inputs (ρ shares)")
    print("=" * 100)
    print(f"{'Run':<40} {'ρ_fb':>10} {'ρ_opt':>10} {'ρ_other':>10}")
    print("-" * 100)
    rho_table = {}
    for name, run in runs.items():
        hist = run.get('history', [])
        if not hist:
            continue
        rho_fb = np.mean([h.get('rho_fb', 0) for h in hist])
        rho_opt = np.mean([h.get('rho_opt', 0) for h in hist])
        rho_other = np.mean([h.get('rho_other', 0) for h in hist])
        print(f"{name:<40} {rho_fb:>10.3f} {rho_opt:>10.3f} {rho_other:>10.3f}")
        rho_table[name] = {'rho_fb': float(rho_fb), 'rho_opt': float(rho_opt),
                            'rho_other': float(rho_other)}
    
    # SC-Dion certificate analysis (if applicable)
    print("\n" + "=" * 100)
    print("SC-Dion certificate diagnostics")
    print("=" * 100)
    for name, run in runs.items():
        stats = run.get('sc_dion_final_stats')
        if stats is None:
            continue
        print(f"\n  {name}:")
        print(f"    pass rate: {stats.get('cert_pass_rate', 0):.3f}")
        print(f"    passes: {stats.get('cert_pass_count', 0):,}")
        print(f"    fails:  {stats.get('cert_fail_count', 0):,}")
        print(f"    mean τ: {stats.get('recent_tau_mean', 0):.3f}")
        print(f"    mean ω: {stats.get('recent_omega_mean', 0):.3f}")
        print(f"    mean cert: {stats.get('recent_cert_mean', 0):+.3f}")
    
    # Final verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    
    wins = []
    for target, td in target_table.items():
        if not td.get('baseline_reached', True):
            continue
        for opt_name, info in td.get('others', {}).items():
            if info.get('reached') and info.get('time_ratio_vs_baseline', 1) < 0.98:
                wins.append((target, opt_name, info['time_ratio_vs_baseline']))
    
    if wins:
        print("\n🏆 Wall-clock wins over baseline:")
        for target, name, ratio in wins:
            saving = (1 - ratio) * 100
            print(f"  • {name} reached loss {target} in {ratio:.3f}× baseline time "
                  f"({saving:.1f}% faster)")
    else:
        print("\n📉 No wall-clock wins detected.")
        print("   Possible reasons:")
        print("   - Model/scale too small (ρ_opt small → no room to improve)")
        print("   - SC-Dion certificate failed too often → fell back to Muon")
        print("   - Convergence slowdown (T_D/T_M) offset compute savings")
        print("   Check: 'ρ_opt' column above. If < 0.10, scale is too small.")
        print("   Check: SC-Dion 'pass rate' above. If < 0.3, structure absent.")
    
    # Save
    out = {
        'baseline': baseline_name,
        'summary': summary,
        'target_table': {str(k): v for k, v in target_table.items()},
        'rho_table': rho_table,
        'wins': [{'target': t, 'optimizer': n, 'ratio': r} for t, n, r in wins],
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n📊 Saved comparison to {args.output}")


if __name__ == '__main__':
    main()
