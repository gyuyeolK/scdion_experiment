"""
Fair eval 분석.

학습 도중 측정된 eval_loss (고정 batches) 기반으로 공정 비교.
이는 train loss EMA보다 robust: 데이터 순서 영향 X.

사용:
    python analysis/fair_eval_analysis.py runs_17b_fair/
"""
import argparse
import json
import re
from pathlib import Path


def load(d: Path):
    f = d / 'history.json'
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def parse_name(name: str):
    """run name → (optimizer, alpha)."""
    m = re.match(r'(\w+?)_a([\d.]+)_fair', name)
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    args = parser.parse_args()
    
    root = Path(args.root)
    runs = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        r = load(d)
        if r:
            runs[d.name] = r
    
    if not runs:
        print("No runs found")
        return
    
    print("="*85)
    print("FAIR EVAL ANALYSIS (eval on identical fixed batches)")
    print("="*85)
    
    # Baseline = Muon
    muon = None
    muon_key = None
    for k in runs:
        if k.startswith('muon'):
            muon = runs[k]
            muon_key = k
            break
    
    if not muon:
        print("No Muon baseline")
        return
    
    muon_step = muon.get('step_time_trimmed_median_ms', 0)
    muon_init = muon.get('initial_eval_loss', 0)
    muon_final = muon.get('final_eval_loss', 0)
    
    print(f"\nBaseline (Muon): step {muon_step:.1f} ms")
    print(f"  eval_loss: {muon_init:.4f} → {muon_final:.4f}  (Δ={muon_final-muon_init:+.4f})")
    
    # ===== Eval loss trajectory comparison =====
    print(f"\n━━━ Eval loss trajectory ━━━")
    
    # 모든 run의 eval step grid를 합치기
    all_steps = set()
    for r in runs.values():
        for h in r.get('history', []):
            all_steps.add(h['step'])
    common_steps = sorted(all_steps)
    
    # Table header
    print(f"  {'Step':<8}", end="")
    for name in sorted(runs.keys()):
        print(f"{name[:20]:<22}", end="")
    print()
    print("  " + "-"*(8 + 22*len(runs)))
    
    for step in common_steps:
        print(f"  {step:<8}", end="")
        for name in sorted(runs.keys()):
            r = runs[name]
            eval_loss = None
            for h in r.get('history', []):
                if h['step'] == step:
                    eval_loss = h.get('eval_loss')
                    break
            if eval_loss is not None:
                print(f"{eval_loss:<22.4f}", end="")
            else:
                print(f"{'-':<22}", end="")
        print()
    
    # ===== Final eval loss + step time =====
    print(f"\n━━━ Final eval_loss and wall-clock ━━━")
    print(f"  {'Run':<25} {'step ms':<10} {'init eval':<12} {'final eval':<12} {'Δeval':<10} {'vs Muon':<12}")
    print("  " + "-"*85)
    
    for name in sorted(runs.keys()):
        r = runs[name]
        step = r.get('step_time_trimmed_median_ms', 0)
        init_e = r.get('initial_eval_loss', 0)
        final_e = r.get('final_eval_loss', 0)
        delta_e = final_e - init_e
        ratio = step / muon_step if muon_step else 0
        eval_diff = final_e - muon_final
        
        ratio_str = "(baseline)" if name == muon_key else f"{ratio:.3f}x"
        eval_diff_str = "(baseline)" if name == muon_key else f"{eval_diff:+.4f}"
        
        print(f"  {name[:23]:<23} {step:<10.1f} {init_e:<12.4f} {final_e:<12.4f} "
              f"{delta_e:<10.4f} {ratio_str}, e:{eval_diff_str}")
    
    # ===== Verdict =====
    print(f"\n━━━ FAIR VERDICT ━━━")
    
    for name in sorted(runs.keys()):
        if name == muon_key:
            continue
        r = runs[name]
        step = r.get('step_time_trimmed_median_ms', 0)
        final_e = r.get('final_eval_loss', 0)
        ratio = step / muon_step if muon_step else 1
        eval_diff = final_e - muon_final
        
        is_faster = ratio < 0.97
        # Eval loss criterion (더 엄격하게: 0.02 = ~1% of typical loss)
        is_loss_ok = abs(eval_diff) < 0.02
        
        if is_faster and is_loss_ok:
            print(f"  🏆 {name}: {(1-ratio)*100:+.1f}% faster, "
                  f"eval Δ={eval_diff:+.4f} → CLEAN WIN")
        elif is_faster and abs(eval_diff) < 0.05:
            print(f"  ✅ {name}: {(1-ratio)*100:+.1f}% faster, "
                  f"eval Δ={eval_diff:+.4f} → slight loss diff but acceptable")
        elif is_faster:
            print(f"  ⚠️ {name}: {(1-ratio)*100:+.1f}% faster but "
                  f"eval Δ={eval_diff:+.4f} → quality trade-off")
        else:
            print(f"  ❌ {name}: ratio {ratio:.3f}x, eval Δ={eval_diff:+.4f}")
    
    # ===== SC-Dion stats summary =====
    print(f"\n━━━ SC-Dion stats by α ━━━")
    for name in sorted(runs.keys()):
        if 'sc_dion' not in name:
            continue
        stats = runs[name].get('sc_dion_final_stats')
        if not stats:
            continue
        opt, alpha = parse_name(name)
        print(f"  α={alpha}: pass {stats.get('cert_pass_rate', 0):.3f}, "
              f"sc_frac {stats.get('sc_dion_fraction', 0):.3f}, "
              f"τ={stats.get('recent_tau_mean', 0):.3f}, "
              f"ω={stats.get('recent_omega_mean', 0):.3f}, "
              f"fallbacks={stats.get('fallback_steps', 0)}")


if __name__ == '__main__':
    main()
