"""
Alpha scaling 분석.

α = 0.5, 0.25, 0.125에서 SC-Dion wall-clock ratio가 어떻게 변하는지.

이론 모델 (논문 Wall-clock identity):
    ratio(α) = ρ_fb + α² · ρ_flop + α · ρ_byte

ρ_fb, ρ_flop, ρ_byte를 측정값으로 fitting하고 예측 vs 실측 비교.

사용:
    python analysis/alpha_scaling.py runs_17b_alpha/
"""
import argparse
import json
import re
from pathlib import Path
import statistics


def load_history(d: Path):
    f = d / 'history.json'
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def parse_alpha(name: str):
    """run name에서 α 추출."""
    m = re.search(r'a([\d.]+)', name)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    args = parser.parse_args()
    
    root = Path(args.root)
    runs = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() and not d.is_symlink():
            continue
        if not (d / 'history.json').exists() and not (d.resolve() / 'history.json').exists():
            continue
        h = load_history(d.resolve() if d.is_symlink() else d)
        if h:
            runs[d.name] = h
    
    if not runs:
        print("No runs found")
        return
    
    # Baseline (Muon)
    muon = None
    for n, r in runs.items():
        if 'muon' in n and 'dion' not in n:
            muon = r
            muon_name = n
            break
    if muon is None:
        print("No Muon baseline found")
        return
    
    muon_step = muon.get('step_time_trimmed_median_ms', 0)
    muon_loss = muon.get('final_loss_ema', 0)
    print("="*80)
    print(f"ALPHA SCALING ANALYSIS")
    print(f"Baseline (Muon): {muon_step:.1f} ms / step, loss {muon_loss:.4f}")
    print("="*80)
    
    # Per-α data points
    sc_dion_points = []  # (α, step_time, ratio, loss_diff)
    dion2_points = []
    
    for name, r in runs.items():
        if name == muon_name:
            continue
        alpha = parse_alpha(name)
        if alpha is None:
            continue
        step = r.get('step_time_trimmed_median_ms', 0)
        loss = r.get('final_loss_ema', 0)
        ratio = step / muon_step if muon_step else 0
        loss_diff = loss - muon_loss
        
        if 'sc_dion' in name:
            sc_dion_points.append((alpha, step, ratio, loss_diff, name))
        elif 'dion2' in name:
            dion2_points.append((alpha, step, ratio, loss_diff, name))
    
    sc_dion_points.sort(key=lambda x: -x[0])  # descending α
    dion2_points.sort(key=lambda x: -x[0])
    
    # ========== Empirical data ==========
    print(f"\n━━━ SC-Dion: α → wall-clock ratio ━━━")
    print(f"{'α':<10} {'step ms':<12} {'ratio':<10} {'savings':<12} {'loss diff':<12}")
    print("-"*60)
    for alpha, step, ratio, ld, _ in sc_dion_points:
        savings = (1 - ratio) * 100
        print(f"  {alpha:<8} {step:<12.1f} {ratio:<10.3f} "
              f"{savings:>+5.1f}%      {ld:>+.4f}")
    
    if dion2_points:
        print(f"\n━━━ Dion2-uniform: α → wall-clock ratio ━━━")
        print(f"{'α':<10} {'step ms':<12} {'ratio':<10} {'savings':<12}")
        print("-"*50)
        for alpha, step, ratio, ld, _ in dion2_points:
            savings = (1 - ratio) * 100
            print(f"  {alpha:<8} {step:<12.1f} {ratio:<10.3f} {savings:>+5.1f}%")
    
    # ========== Theoretical model fit ==========
    # ratio(α) = ρ_fb + ρ_flop·α² + ρ_byte·α
    # 가정: single-GPU, ρ_byte ≈ 0
    # ratio(α) ≈ ρ_fb + ρ_flop·α²
    #
    # ρ_fb + ρ_flop = 1 (α=1일 때 == Muon == ratio 1)
    # → ρ_flop = (1 - ratio(α)) / (1 - α²)
    
    if len(sc_dion_points) >= 2:
        print(f"\n━━━ Theoretical model fit ━━━")
        print(f"  Model: ratio(α) = ρ_fb + ρ_flop·α²  (assuming ρ_byte=0, single GPU)")
        print(f"")
        
        # Fit ρ_flop from each data point
        rho_flop_estimates = []
        for alpha, step, ratio, _, _ in sc_dion_points:
            if alpha < 1.0:
                rho_flop = (1 - ratio) / (1 - alpha**2)
                rho_flop_estimates.append((alpha, rho_flop))
                print(f"  α={alpha}: implied ρ_flop = {rho_flop:.3f}")
        
        if rho_flop_estimates:
            rho_flop_mean = sum(r[1] for r in rho_flop_estimates) / len(rho_flop_estimates)
            rho_fb = 1 - rho_flop_mean
            print(f"\n  Mean ρ_flop: {rho_flop_mean:.3f}")
            print(f"  Implied ρ_fb: {rho_fb:.3f}  (forward+backward time fraction)")
            
            print(f"\n  Floor (α→0): {rho_fb:.3f}")
            print(f"  → SC-Dion이 α→0으로 가도 ρ_fb 아래로는 못 내려감")
            print(f"  → 최대 가능 절감: {(1-rho_fb)*100:.1f}%")
            
            # Predicted vs actual
            print(f"\n━━━ Predicted vs Actual ━━━")
            print(f"  {'α':<10} {'predicted':<12} {'actual':<12} {'error':<10}")
            print("  " + "-"*50)
            for alpha, _, ratio, _, _ in sc_dion_points:
                predicted = rho_fb + rho_flop_mean * alpha**2
                error = (predicted - ratio) / ratio * 100
                print(f"  {alpha:<8} {predicted:<12.3f} {ratio:<12.3f} {error:>+5.1f}%")
    
    # ========== Trend verification ==========
    print(f"\n━━━ Scaling trend verification ━━━")
    if len(sc_dion_points) >= 2:
        # α=0.5 → α=0.25: ratio change?
        for i in range(len(sc_dion_points) - 1):
            a1, _, r1, _, _ = sc_dion_points[i]
            a2, _, r2, _, _ = sc_dion_points[i+1]
            
            ratio_delta = r1 - r2  # 음수면 줄어드는 alpha가 더 빠름
            # α² 이론: (r1 - ρ_fb) / (r2 - ρ_fb) = (a1/a2)²
            if 'rho_fb' in dir():
                pass
            print(f"  α {a1} → {a2}: ratio {r1:.3f} → {r2:.3f}  "
                  f"(Δ = {ratio_delta:+.3f})")
            if ratio_delta > 0:
                print(f"    ✅ Smaller α is faster (as theory predicts)")
            else:
                print(f"    ⚠️ Smaller α is NOT faster (theory contradicted)")
    
    # ========== Loss preservation ==========
    print(f"\n━━━ Loss preservation across α ━━━")
    all_losses_ok = True
    for alpha, _, _, ld, _ in sc_dion_points:
        marker = "✅" if abs(ld) < 0.01 else ("⚠️" if abs(ld) < 0.05 else "❌")
        print(f"  α={alpha}: Δloss = {ld:+.4f}  {marker}")
        if abs(ld) >= 0.05:
            all_losses_ok = False
    
    # ========== Final summary ==========
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    
    if not sc_dion_points:
        return
    
    best_alpha, _, best_ratio, _, _ = min(sc_dion_points, key=lambda x: x[2])
    print(f"\n🏆 Best α: {best_alpha} → {(1-best_ratio)*100:.1f}% faster than Muon")
    
    # Check if ratio is monotonically decreasing with α
    is_monotonic = all(sc_dion_points[i][2] >= sc_dion_points[i+1][2]
                       for i in range(len(sc_dion_points)-1))
    if is_monotonic:
        print(f"📈 Wall-clock ratio decreases monotonically with α (theory ✅)")
    else:
        print(f"📉 Wall-clock ratio NOT monotonic with α (unexpected)")
    
    if all_losses_ok:
        print(f"✅ Loss preserved across all α (no quality degradation)")
    
    # α² scaling check (rough)
    if len(sc_dion_points) >= 2 and 'rho_fb' in dir():
        # Are residuals (ratio - rho_fb) scaling as α²?
        # Compute (r - rho_fb) / α² and see if approx constant
        const_check = []
        for alpha, _, ratio, _, _ in sc_dion_points:
            if alpha > 0:
                residual = ratio - rho_fb
                const = residual / (alpha ** 2)
                const_check.append((alpha, const))
        
        constants = [c for _, c in const_check]
        if constants and max(constants) / min(constants) < 1.3:
            print(f"✅ α² scaling holds: residuals/(α²) ≈ const")
            print(f"   Values: {', '.join(f'{c:.3f}' for _, c in const_check)}")
        else:
            print(f"⚠️ α² scaling imperfect: residuals/(α²) varies")
            print(f"   Values: {', '.join(f'{c:.3f}' for _, c in const_check)}")


if __name__ == '__main__':
    main()
