"""
Phase 2 (small) 결과 비교 분석.

세 옵티마이저 run을 받아서 step time, loss curve, SC-Dion 통계 비교.

사용:
    python analysis/analyze_phase2_small.py runs_small/muon_a0.5 runs_small/dion2_uniform_a0.5 runs_small/sc_dion_a0.5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_run(p: Path) -> dict:
    f = p / 'history.json'
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', type=str, nargs='+')
    args = parser.parse_args()
    
    runs = {}
    for d in args.run_dirs:
        p = Path(d)
        r = load_run(p)
        if r is None:
            print(f"  [skip] {d}: no history.json")
            continue
        runs[p.name] = r
    
    if not runs:
        print("No valid runs to compare!")
        sys.exit(1)
    
    # Identify baseline (Muon)
    baseline_name = None
    for n in runs:
        if 'muon' in n.lower() and 'dion' not in n.lower():
            baseline_name = n
            break
    if baseline_name is None:
        baseline_name = list(runs.keys())[0]
    
    print("="*90)
    print(f"COMPARISON  (baseline: {baseline_name})")
    print("="*90)
    
    # ===== Step time comparison =====
    print("\n━━━ Step time (smaller is better) ━━━")
    print(f"{'Run':<35} {'median ms':<12} {'trimmed ms':<12} {'mean ms':<12} {'vs baseline':<14}")
    print("-"*85)
    
    base_median = None
    for name, r in runs.items():
        med = r.get('step_time_median_ms', np.median(r.get('step_times', [0])))
        trim = r.get('step_time_trimmed_median_ms', med)
        mean = r.get('step_time_mean_ms', np.mean(r.get('step_times', [0])))
        if name == baseline_name:
            base_median = trim
            ratio_str = "(baseline)"
        elif base_median:
            ratio = trim / base_median
            ratio_str = f"{ratio:.3f}x"
        else:
            ratio_str = "?"
        print(f"  {name[:33]:<33} {med:<12.1f} {trim:<12.1f} {mean:<12.1f} {ratio_str:<14}")
    
    # ===== Total wall-clock =====
    print("\n━━━ Total wall-clock ━━━")
    print(f"{'Run':<35} {'Total (s)':<12} {'Steps':<8} {'ms/step (avg)':<14}")
    print("-"*70)
    base_total = None
    for name, r in runs.items():
        total = r.get('total_elapsed_sec', 0)
        n_steps = len(r.get('step_times', []))
        avg = total / n_steps * 1000 if n_steps else 0
        if name == baseline_name:
            base_total = total
            ratio_str = "(baseline)"
        elif base_total:
            ratio_str = f"{total/base_total:.3f}x"
        else:
            ratio_str = "?"
        print(f"  {name[:33]:<33} {total:<12.1f} {n_steps:<8} {avg:<14.1f} {ratio_str}")
    
    # ===== Loss curve =====
    print("\n━━━ Loss progression (loss_ema sampled at intervals) ━━━")
    
    # 공통 step grid 가져오기
    all_history_steps = set()
    for r in runs.values():
        for h in r.get('history', []):
            all_history_steps.add(h['step'])
    common_steps = sorted(all_history_steps)
    
    if common_steps:
        # Table header
        header = f"  {'Step':<8}"
        for name in runs:
            header += f"{name[:18]:<20}"
        print(header)
        print("  " + "-"*(8 + 20*len(runs)))
        
        for step in common_steps[::2]:  # 매번 다 보이면 길어서 격step
            row = f"  {step:<8}"
            for name, r in runs.items():
                # find matching history entry
                loss = None
                for h in r.get('history', []):
                    if h['step'] == step:
                        loss = h.get('loss_ema', h.get('loss'))
                        break
                if loss is not None:
                    row += f"{loss:<20.4f}"
                else:
                    row += f"{'-':<20}"
            print(row)
    
    # Final loss
    print(f"\n━━━ Final loss (EMA) ━━━")
    print(f"  {'Run':<35} {'Loss EMA':<12} {'vs baseline':<14}")
    base_loss = None
    for name, r in runs.items():
        l = r.get('final_loss_ema', 0)
        if name == baseline_name:
            base_loss = l
            ratio_str = "(baseline)"
        elif base_loss:
            ratio_str = f"{l - base_loss:+.4f}"
        else:
            ratio_str = "?"
        print(f"  {name[:33]:<33} {l:<12.4f} {ratio_str}")
    
    # ===== SC-Dion specific =====
    print(f"\n━━━ SC-Dion specific stats ━━━")
    for name, r in runs.items():
        stats = r.get('sc_dion_final_stats')
        if not stats:
            continue
        print(f"  {name}:")
        print(f"    Cert pass rate:     {stats.get('cert_pass_rate', 0):.3f}")
        print(f"    SC-Dion fraction:   {stats.get('sc_dion_fraction', 0):.3f}")
        print(f"    Cert evals:         {stats.get('cert_eval_count', 0)}")
        print(f"    SC-Dion steps:      {stats.get('sc_dion_steps', 0)}")
        print(f"    Fallback steps:     {stats.get('fallback_steps', 0)}")
        if 'recent_tau_mean' in stats:
            print(f"    Recent mean τ:      {stats['recent_tau_mean']:.3f}")
        if 'recent_omega_mean' in stats:
            print(f"    Recent mean ω:      {stats['recent_omega_mean']:.3f}")
    
    # ===== Verdict =====
    print(f"\n{'='*90}")
    print("VERDICT")
    print(f"{'='*90}\n")
    
    # Was there a wall-clock win for any non-baseline?
    base_r = runs.get(baseline_name)
    base_trim = base_r.get('step_time_trimmed_median_ms', 0) if base_r else 0
    base_final_loss = base_r.get('final_loss_ema', 0) if base_r else 0
    
    print("Wall-clock per step (the key metric):")
    for name, r in runs.items():
        if name == baseline_name:
            continue
        trim = r.get('step_time_trimmed_median_ms', 0)
        ratio = trim / base_trim if base_trim else 0
        if ratio < 0.85:
            print(f"  ✅ {name}: {ratio:.3f}x baseline — meaningful step time savings")
        elif ratio < 0.98:
            print(f"  🟢 {name}: {ratio:.3f}x baseline — modest step time savings")
        elif ratio < 1.05:
            print(f"  🟡 {name}: {ratio:.3f}x baseline — comparable")
        else:
            print(f"  🔴 {name}: {ratio:.3f}x baseline — slower than Muon")
    
    print("\nLoss after equal step count:")
    for name, r in runs.items():
        if name == baseline_name:
            continue
        final_loss = r.get('final_loss_ema', 0)
        loss_diff = final_loss - base_final_loss
        if loss_diff < 0.01:
            print(f"  ✅ {name}: Δloss = {loss_diff:+.4f} — matches Muon's convergence")
        elif loss_diff < 0.05:
            print(f"  🟢 {name}: Δloss = {loss_diff:+.4f} — slight degradation")
        elif loss_diff < 0.15:
            print(f"  🟡 {name}: Δloss = {loss_diff:+.4f} — moderate degradation")
        else:
            print(f"  🔴 {name}: Δloss = {loss_diff:+.4f} — significant degradation")
    
    print("\nCombined verdict (need BOTH faster + similar loss):")
    for name, r in runs.items():
        if name == baseline_name:
            continue
        trim = r.get('step_time_trimmed_median_ms', 0)
        final_loss = r.get('final_loss_ema', 0)
        time_ratio = trim / base_trim if base_trim else 1
        loss_diff = final_loss - base_final_loss
        
        # 전체 wall-clock으로 same-loss 도달 시간 추정 (heuristic):
        # time = step_count * step_time. step_count to same loss는 모르지만,
        # final loss가 비슷하면 step_time 절감이 그대로 wall-clock 이득.
        if time_ratio < 0.95 and loss_diff < 0.05:
            print(f"  🏆 {name}: WALL-CLOCK WIN — {(1-time_ratio)*100:.1f}% faster with comparable loss")
        elif time_ratio < 0.95 and loss_diff < 0.15:
            print(f"  ⚖️  {name}: TRADE-OFF — faster but with some loss degradation")
        elif time_ratio < 0.95:
            print(f"  ❌ {name}: FASTER but SIGNIFICANT loss degradation")
        else:
            print(f"  ❌ {name}: NO WIN — not faster than Muon")


if __name__ == '__main__':
    main()
