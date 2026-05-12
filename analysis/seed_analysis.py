"""
시드 robustness 분석.

여러 시드에서 같은 (optimizer, alpha) 조합을 학습한 결과를 받아서:
- 각 optimizer에 대해 step time, loss의 mean ± std 계산
- Muon 대비 wall-clock ratio의 분포 확인
- 결과가 robust한지 verdict

사용:
    python analysis/seed_analysis.py runs_17b_seeds/
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load_runs(root: Path):
    """root 안의 모든 history.json 파일을 읽어서 {(optimizer, seed): data} 로 구성."""
    pattern = re.compile(r'(.+)_s(\d+)$')
    runs = {}
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        hist = sub / 'history.json'
        if not hist.exists():
            continue
        m = pattern.match(sub.name)
        if not m:
            continue
        opt, seed = m.group(1), int(m.group(2))
        with open(hist) as f:
            runs[(opt, seed)] = json.load(f)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='Directory with optimizer_sN/ subdirs')
    args = parser.parse_args()
    
    runs = load_runs(Path(args.root))
    if not runs:
        print(f"No runs found in {args.root}")
        return
    
    # Organize by optimizer
    by_opt = defaultdict(dict)  # {opt: {seed: data}}
    for (opt, seed), data in runs.items():
        by_opt[opt][seed] = data
    
    print("="*90)
    print(f"SEED ROBUSTNESS ANALYSIS")
    print(f"Root: {args.root}")
    print("="*90)
    
    # ===== Per-optimizer stats =====
    print(f"\n━━━ Per-optimizer results across seeds ━━━")
    
    summary = {}
    for opt in sorted(by_opt.keys()):
        seeds = sorted(by_opt[opt].keys())
        step_times = [by_opt[opt][s].get('step_time_trimmed_median_ms', 0) for s in seeds]
        final_losses = [by_opt[opt][s].get('final_loss_ema', 0) for s in seeds]
        totals = [by_opt[opt][s].get('total_elapsed_sec', 0) for s in seeds]
        
        n = len(seeds)
        step_mean = sum(step_times) / n
        step_std = statistics.stdev(step_times) if n > 1 else 0
        loss_mean = sum(final_losses) / n
        loss_std = statistics.stdev(final_losses) if n > 1 else 0
        total_mean = sum(totals) / n
        
        summary[opt] = {
            'seeds': seeds,
            'step_times': step_times,
            'final_losses': final_losses,
            'step_mean': step_mean, 'step_std': step_std,
            'loss_mean': loss_mean, 'loss_std': loss_std,
            'total_mean': total_mean,
        }
        
        print(f"\n  {opt} (n={n} seeds: {seeds}):")
        print(f"    Step time:    {step_mean:.1f} ± {step_std:.1f} ms")
        print(f"      individual: {', '.join(f'{t:.1f}' for t in step_times)}")
        print(f"    Final loss:   {loss_mean:.4f} ± {loss_std:.4f}")
        print(f"      individual: {', '.join(f'{l:.4f}' for l in final_losses)}")
        print(f"    Total time:   {total_mean:.1f} s (avg)")
    
    # ===== Ratio vs baseline (muon) =====
    print(f"\n━━━ Ratio vs Muon (cross-seed) ━━━")
    if 'muon' not in summary:
        print("  No muon baseline found")
        return
    
    muon_stats = summary['muon']
    muon_seeds = set(muon_stats['seeds'])
    
    for opt in sorted(by_opt.keys()):
        if opt == 'muon':
            continue
        opt_stats = summary[opt]
        opt_seeds = set(opt_stats['seeds'])
        common_seeds = sorted(muon_seeds & opt_seeds)
        
        if not common_seeds:
            print(f"\n  {opt}: no common seeds with Muon")
            continue
        
        # Per-seed ratio
        ratios = []
        loss_diffs = []
        for seed in common_seeds:
            muon_t = by_opt['muon'][seed].get('step_time_trimmed_median_ms', 0)
            opt_t = by_opt[opt][seed].get('step_time_trimmed_median_ms', 0)
            if muon_t > 0:
                ratios.append(opt_t / muon_t)
            muon_l = by_opt['muon'][seed].get('final_loss_ema', 0)
            opt_l = by_opt[opt][seed].get('final_loss_ema', 0)
            loss_diffs.append(opt_l - muon_l)
        
        n = len(ratios)
        r_mean = sum(ratios) / n
        r_std = statistics.stdev(ratios) if n > 1 else 0
        r_min, r_max = min(ratios), max(ratios)
        l_mean = sum(loss_diffs) / n
        l_std = statistics.stdev(loss_diffs) if n > 1 else 0
        
        print(f"\n  {opt} (n={n} matched seeds):")
        print(f"    Wall-clock ratio:  {r_mean:.3f} ± {r_std:.3f}  (range: {r_min:.3f}-{r_max:.3f})")
        print(f"      per seed: " + ", ".join(
            f"s{s}={r:.3f}" for s, r in zip(common_seeds, ratios)
        ))
        print(f"    Loss difference:   {l_mean:+.4f} ± {l_std:.4f}")
        print(f"      per seed: " + ", ".join(
            f"s{s}={d:+.4f}" for s, d in zip(common_seeds, loss_diffs)
        ))
    
    # ===== Robustness verdict =====
    print(f"\n{'='*90}")
    print("ROBUSTNESS VERDICT")
    print(f"{'='*90}")
    
    for opt in sorted(by_opt.keys()):
        if opt == 'muon':
            continue
        if opt not in summary:
            continue
        
        opt_seeds = set(summary[opt]['seeds'])
        common_seeds = sorted(muon_seeds & opt_seeds)
        if len(common_seeds) < 2:
            print(f"\n  {opt}: need at least 2 seeds for robustness assessment")
            continue
        
        ratios = []
        loss_diffs = []
        for seed in common_seeds:
            muon_t = by_opt['muon'][seed].get('step_time_trimmed_median_ms', 0)
            opt_t = by_opt[opt][seed].get('step_time_trimmed_median_ms', 0)
            if muon_t > 0:
                ratios.append(opt_t / muon_t)
            muon_l = by_opt['muon'][seed].get('final_loss_ema', 0)
            opt_l = by_opt[opt][seed].get('final_loss_ema', 0)
            loss_diffs.append(opt_l - muon_l)
        
        r_mean = sum(ratios) / len(ratios)
        r_std = statistics.stdev(ratios) if len(ratios) > 1 else 0
        l_mean = sum(loss_diffs) / len(loss_diffs)
        l_std_abs = statistics.stdev([abs(d) for d in loss_diffs]) if len(loss_diffs) > 1 else 0
        
        # Coefficient of variation (CV) — smaller is more reproducible
        cv = r_std / r_mean if r_mean > 0 else 999
        
        print(f"\n  {opt}:")
        print(f"    Mean wall-clock ratio:  {r_mean:.3f}")
        print(f"    Std across seeds:        {r_std:.3f}")
        print(f"    CV (std/mean):           {cv:.3f}  ", end='')
        if cv < 0.03:
            print("⭐ Very reproducible")
        elif cv < 0.08:
            print("✅ Reproducible")
        elif cv < 0.15:
            print("⚠️ Moderately variable")
        else:
            print("❌ High variance — not robust")
        
        # Loss reproducibility (all seeds should give similar final loss)
        print(f"    Loss diff mean:          {l_mean:+.4f}")
        if abs(l_mean) < 0.005 and l_std_abs < 0.005:
            print(f"    Loss match:              ✅ Indistinguishable from Muon")
        elif abs(l_mean) < 0.02:
            print(f"    Loss match:              ✅ Within noise (acceptable)")
        else:
            print(f"    Loss match:              ⚠️ Some degradation")
        
        # Overall
        is_faster = r_mean < 0.95
        is_loss_ok = abs(l_mean) < 0.02
        is_robust = cv < 0.10
        
        print(f"\n    Final verdict: ", end='')
        if is_faster and is_loss_ok and is_robust:
            print(f"🏆 ROBUST WALL-CLOCK WIN")
            print(f"       → {(1-r_mean)*100:.1f}% faster, same loss, reproducible across seeds")
        elif is_faster and is_loss_ok:
            print(f"✅ Wall-clock win, but high seed variance")
        elif is_faster:
            print(f"⚠️ Faster but loss degradation")
        elif is_loss_ok and is_robust:
            print(f"🟡 Comparable to Muon (no clear win)")
        else:
            print(f"❌ No clear win")
    
    # ===== Combined report (mean across all seeds incl. orig 42 if present) =====
    print(f"\n{'='*90}")
    print("COMBINED REPORT (mean across all seeds)")
    print(f"{'='*90}")
    
    print(f"\n{'Optimizer':<25} {'n':<5} {'mean step ms':<15} {'mean loss':<12} {'ratio vs Muon':<15}")
    print("-"*75)
    muon_mean_step = summary['muon']['step_mean']
    for opt in sorted(summary.keys()):
        s = summary[opt]
        ratio = s['step_mean'] / muon_mean_step if opt != 'muon' else 1.0
        ratio_str = "(baseline)" if opt == 'muon' else f"{ratio:.3f}x"
        print(f"  {opt:<23} {len(s['seeds']):<5} "
              f"{s['step_mean']:.1f} ± {s['step_std']:.1f}    "
              f"{s['loss_mean']:.4f}     "
              f"{ratio_str}")


if __name__ == '__main__':
    main()
