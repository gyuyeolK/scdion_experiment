"""
Quick sanity check on a coverable matrix recovery toy problem.
논문의 Table 7 (App.~\ref{app:sc-dion-toy})을 small-scale로 재현.

이 테스트는 SC-Dion이 실제로 작동하는지 검증함 (CPU에서도 1분 내 끝남).
실제 GPU + LLM 실험 시작 전에 코드 sanity check로 사용.

사용:
    python scripts/sanity_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from optimizers import Muon, Dion2Uniform, SCDion


def make_coverable_target(m=128, n=128, rank=4, support_size=12, seed=0):
    """Rank-r target supported on `support_size` rows. SC-Dion에 유리한 구조."""
    g = torch.Generator().manual_seed(seed)
    U = torch.zeros(m, rank)
    # First `support_size` rows have nonzero entries
    U[:support_size, :] = torch.randn(support_size, rank, generator=g)
    V = torch.randn(n, rank, generator=g)
    sigma = torch.diag(torch.linspace(1.0, 0.5, rank))
    W_star = U @ sigma @ V.t()
    return W_star


def run_one(opt_name: str, W_star: torch.Tensor, alpha: float = 0.25,
            lr: float = 0.3, max_steps: int = 300, target: float = 0.1):
    """Return (steps to target rel-err, final rel-err, flops_per_step_proxy)"""
    W = torch.zeros_like(W_star, requires_grad=True)
    
    m, n = W.shape
    
    if opt_name == 'muon':
        opt = Muon([W], lr=lr, momentum=0.95, ns_steps=5)
        flops_factor = 1.0
    elif opt_name == 'dion2_uniform':
        opt = Dion2Uniform([W], lr=lr, alpha=alpha, mu=0.95, ns_steps=5)
        flops_factor = alpha ** 2
    elif opt_name == 'sc_dion':
        opt = SCDion([W], lr=lr, alpha_u=alpha, alpha_d=1.0, mu=0.95,
                     subspace_rank=4, cert_threshold=0.05, refresh_period=5)
        flops_factor = alpha ** 2
    else:
        raise ValueError(opt_name)
    
    target_step = None
    initial_err = float(torch.norm(W_star).item())
    for step in range(max_steps):
        opt.zero_grad()
        loss = 0.5 * ((W - W_star) ** 2).sum()
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            rel_err = float(torch.norm(W - W_star).item()) / initial_err
        
        if target_step is None and rel_err <= target:
            target_step = step + 1
            break
    
    final_err = rel_err
    cert_info = ""
    if opt_name == 'sc_dion':
        cs = opt.get_cert_stats()
        cert_info = f" cert_pass={cs['cert_pass_rate']:.2f}"
    
    return target_step, final_err, flops_factor, cert_info


def main():
    print("=" * 80)
    print("SANITY CHECK: Coverable matrix recovery (paper's Table 7 mini version)")
    print("=" * 80)
    print()
    
    # Setup
    W_star = make_coverable_target(m=128, n=128, rank=4, support_size=12, seed=42)
    print(f"Target: shape {W_star.shape}, rank 4, supported on 12 rows")
    print(f"Target rel-err: 0.1")
    print()
    
    print(f"{'Optimizer':<20} {'α':>6} {'Steps':>8} {'Final err':>10} {'Compute/Muon':>14} {'Notes':<20}")
    print("-" * 80)
    
    # Muon baseline
    s, e, f, _ = run_one('muon', W_star)
    muon_steps = s if s else 300
    print(f"{'muon':<20} {1.0:>6.2f} {s if s else 'n.r.':>8} {e:>10.4f} {1.0:>14.3f}")
    
    # Sweeps
    for alpha in [0.5, 0.25, 0.125]:
        for opt_name in ['dion2_uniform', 'sc_dion']:
            s, e, f, ci = run_one(opt_name, W_star, alpha=alpha)
            step_disp = str(s) if s else 'n.r.'
            ratio = s / muon_steps if s else float('nan')
            total_compute = f * (s / muon_steps) if s else float('inf')
            print(f"{opt_name:<20} {alpha:>6.3f} {step_disp:>8} {e:>10.4f} "
                  f"{total_compute:>14.3f}{ci}")
    
    print()
    print("Expected from paper Table 7 (App. F.5):")
    print("  - muon: ~100 steps, baseline 1.0×")
    print("  - dion2_uniform α=0.5:   ~170 steps (1.7× Muon's steps), no win")
    print("  - dion2_uniform α=0.25:  ~290 steps (2.8× Muon's steps), no win")  
    print("  - dion2_uniform α=0.125: NOT REACHED (fails on hard case)")
    print("  - sc_dion α=0.5:   ~100 steps (1.0×), compute 0.25  ← STRICT WIN")
    print("  - sc_dion α=0.25:  ~100 steps (1.0×), compute 0.0625 ← STRICT WIN")
    print("  - sc_dion α=0.125: ~100 steps (1.0×), compute 0.0156 ← STRICT WIN")
    print()
    print("If SC-Dion matches Muon in steps while using α² compute, the strict-win")
    print("behavior is verified — paper's Cor 4.3.")


if __name__ == '__main__':
    main()
