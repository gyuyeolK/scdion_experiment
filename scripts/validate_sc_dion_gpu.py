"""
SC-Dion GPU vs CPU 검증.

세 가지 확인:
1. Sanity test (toy problem) - GPU 구현이 CPU와 같은 수렴 패턴 보이는지
2. Numerical agreement - 같은 input에 cert/τ/ω 값이 일치하는지
3. Wall-clock benchmark - 각 함수가 실제로 얼마나 빨라졌는지

사용:
    python scripts/validate_sc_dion_gpu.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from optimizers import Muon, Dion2Uniform, SCDion, SCDionGPU
from optimizers.sc_dion import (
    _randomized_subspace as cpu_subspace,
    _greedy_logdet_select as cpu_select,
    _certificate as cpu_cert,
)
from optimizers.sc_dion_gpu import (
    _randomized_subspace_gpu as gpu_subspace,
    _greedy_logdet_select_gpu as gpu_select,
    _certificate_gpu as gpu_cert,
)


def log(s):
    print(s, flush=True)


def make_coverable_target(m=128, n=128, rank=4, support_size=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.zeros(m, rank)
    U[:support_size, :] = torch.randn(support_size, rank, generator=g)
    V = torch.randn(n, rank, generator=g)
    sigma = torch.diag(torch.linspace(1.0, 0.5, rank))
    return U @ sigma @ V.t()


def numerical_agreement_test():
    """같은 random tensor에 대해 GPU vs CPU 함수 결과가 비슷한지."""
    log("\n" + "="*70)
    log("Test 1: Numerical agreement (GPU vs CPU)")
    log("="*70)
    
    if not torch.cuda.is_available():
        log("  CUDA not available, skipping")
        return
    
    torch.manual_seed(42)
    # Realistic LLM gradient size (rough)
    m, n = 2048, 8192  # FFN W_up scale
    k = 8
    alpha = 0.25
    num_select = int(alpha * m)
    
    log(f"  Test matrix: {m}x{n}, k={k}, num_select={num_select}")
    
    # Same data on CPU and GPU
    S_cpu = torch.randn(m, n)
    S_gpu = S_cpu.cuda()
    
    # Subspace
    torch.manual_seed(42)
    U_cpu, tau_cpu = cpu_subspace(S_cpu, k=k)
    torch.manual_seed(42)
    U_gpu, tau_gpu = gpu_subspace(S_gpu, k=k)
    
    # tau는 GPU에서 tensor로 반환됨
    tau_cpu_val = float(tau_cpu) if not torch.is_tensor(tau_cpu) else float(tau_cpu.item())
    tau_gpu_val = float(tau_gpu.item())
    
    log(f"  τ (CPU):  {tau_cpu_val:.4f}")
    log(f"  τ (GPU):  {tau_gpu_val:.4f}")
    log(f"  τ diff:   {abs(tau_cpu_val - tau_gpu_val):.6f}")
    
    # U_k subspace 자체는 random seed + 다른 randn으로 다를 수 있지만, span은 비슷해야 함
    # subspace similarity: ||U_cpu^T U_gpu||_F^2 / k → 1이면 같은 span
    if U_cpu.shape == U_gpu.shape:
        U_gpu_cpu = U_gpu.cpu()
        # Project U_gpu onto span(U_cpu)
        proj = (U_cpu @ (U_cpu.t() @ U_gpu_cpu))
        sim = ((proj * U_gpu_cpu).sum() / (U_gpu_cpu * U_gpu_cpu).sum()).item()
        log(f"  Subspace alignment (1.0=identical span): {sim:.3f}")
    
    # Selection - 다를 수 있지만 비슷한 logdet quality를 줘야 함
    sel_cpu = cpu_select(U_cpu, num_select)
    sel_gpu = gpu_select(U_gpu, num_select).cpu()
    
    # Selected의 logdet 비교
    def selected_logdet(U, idx):
        U_s = U[idx]
        G = U_s.t() @ U_s + 1e-10 * torch.eye(U_s.size(1))
        sign, logdet = torch.linalg.slogdet(G)
        return float(logdet)
    
    ld_cpu = selected_logdet(U_cpu, sel_cpu)
    ld_gpu = selected_logdet(U_gpu.cpu(), sel_gpu)
    log(f"  Selected logdet (CPU): {ld_cpu:+.3f}")
    log(f"  Selected logdet (GPU): {ld_gpu:+.3f}")
    log(f"  Logdet diff:           {abs(ld_cpu - ld_gpu):.3f}")
    
    # Certificate
    c_cpu, om_cpu = cpu_cert(U_cpu, sel_cpu, tau_cpu_val)
    c_gpu_t, om_gpu_t = gpu_cert(U_gpu, sel_gpu.cuda(), tau_gpu)
    c_gpu = float(c_gpu_t.item())
    om_gpu = float(om_gpu_t.item())
    
    log(f"  cert (CPU): {c_cpu:+.3f}, ω: {om_cpu:.3f}")
    log(f"  cert (GPU): {c_gpu:+.3f}, ω: {om_gpu:.3f}")
    
    if abs(c_cpu - c_gpu) < 0.1:
        log("  ✅ Certificates agree within 0.1 (randomization noise expected)")
    else:
        log("  ⚠️ Certificates differ by more than 0.1; check implementation")


def speed_benchmark():
    """각 함수의 GPU vs CPU 속도."""
    log("\n" + "="*70)
    log("Test 2: Speed benchmark")
    log("="*70)
    
    if not torch.cuda.is_available():
        log("  CUDA not available, skipping")
        return
    
    # Realistic param sizes
    sizes = [
        (2048, 2048, "attn_proj"),  # like attention projection
        (2048, 8192, "ffn_up"),      # like FFN up projection
        (8192, 2048, "ffn_down"),    # like FFN down projection
    ]
    k = 8
    alpha = 0.25
    
    log(f"  k={k}, α={alpha}, NS-style timing on A100\n")
    log(f"  {'Shape':<20} {'Func':<20} {'CPU ms':<12} {'GPU ms':<12} {'Speedup':<10}")
    log("  " + "-"*72)
    
    for m, n, name in sizes:
        num_select = max(1, int(alpha * m))
        S_cpu = torch.randn(m, n)
        S_gpu = S_cpu.cuda()
        
        # ---- _randomized_subspace ----
        # Warmup
        for _ in range(3):
            cpu_subspace(S_cpu, k=k)
            gpu_subspace(S_gpu, k=k)
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(5):
            U_cpu, _ = cpu_subspace(S_cpu, k=k)
        t_cpu_sub = (time.perf_counter() - t0) / 5 * 1000
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            U_gpu, _ = gpu_subspace(S_gpu, k=k)
            torch.cuda.synchronize()
        t_gpu_sub = (time.perf_counter() - t0) / 5 * 1000
        
        log(f"  {f'{m}x{n} ({name})':<20} {'subspace':<20} "
            f"{t_cpu_sub:<12.1f} {t_gpu_sub:<12.1f} {t_cpu_sub/t_gpu_sub:<10.1f}x")
        
        # ---- _greedy_logdet_select ----
        t0 = time.perf_counter()
        for _ in range(5):
            cpu_select(U_cpu, num_select)
        t_cpu_sel = (time.perf_counter() - t0) / 5 * 1000
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            gpu_select(U_gpu, num_select)
            torch.cuda.synchronize()
        t_gpu_sel = (time.perf_counter() - t0) / 5 * 1000
        
        log(f"  {'':<20} {'logdet_select':<20} "
            f"{t_cpu_sel:<12.1f} {t_gpu_sel:<12.1f} {t_cpu_sel/t_gpu_sel:<10.1f}x")
        
        # Total per-param diagnostic time
        log(f"  {'':<20} {'TOTAL':<20} "
            f"{t_cpu_sub+t_cpu_sel:<12.1f} {t_gpu_sub+t_gpu_sel:<12.1f} "
            f"{(t_cpu_sub+t_cpu_sel)/(t_gpu_sub+t_gpu_sel):<10.1f}x")
        log("")


def optimizer_step_benchmark():
    """전체 옵티마이저 step time 비교: Muon vs SCDionGPU."""
    log("\n" + "="*70)
    log("Test 3: Optimizer step time (Muon vs SCDionGPU)")
    log("="*70)
    
    if not torch.cuda.is_available():
        log("  CUDA not available, skipping")
        return
    
    device = 'cuda'
    
    log("\n  Setup: 169 params total (mimicking SmolLM2-1.7B's 2D weights)")
    
    # Mix of shapes (typical LLM)
    params = []
    for _ in range(60):
        params.append(torch.nn.Parameter(torch.randn(2048, 8192, device=device, dtype=torch.bfloat16)))
    for _ in range(80):
        params.append(torch.nn.Parameter(torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)))
    for _ in range(29):
        params.append(torch.nn.Parameter(torch.randn(8192, 2048, device=device, dtype=torch.bfloat16)))
    log(f"    {len(params)} params")
    
    def reset_grads():
        for p in params:
            p.grad = torch.randn_like(p) * 0.01
    
    muon = Muon(params, lr=1e-3)
    sc_dion = SCDionGPU(params, lr=1e-3, alpha_u=0.5, subspace_rank=8,
                        cert_threshold=0.05, refresh_period=20)
    
    # ========== Muon: extended warmup + measurement ==========
    log("  Warmup Muon (10 steps)...")
    for _ in range(10):
        reset_grads()
        muon.step()
    torch.cuda.synchronize()
    
    times_muon = []
    for _ in range(20):
        reset_grads()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        muon.step()
        torch.cuda.synchronize()
        times_muon.append((time.perf_counter() - t0) * 1000)
    
    # ========== SC-Dion: extended warmup (>K to cover one full cycle) ==========
    log("  Warmup SC-Dion (30 steps to cover full refresh cycle)...")
    for _ in range(30):
        reset_grads()
        sc_dion.step()
    torch.cuda.synchronize()
    
    times_sc = []
    is_refresh = []
    # K=20 → measure 40 steps to see 2 full cycles
    for measurement_step in range(40):
        reset_grads()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sc_dion.step()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_sc.append(elapsed_ms)
        # Refresh happens when step_count % K == 0. After 30 warmup steps,
        # step_count starts at 30 → 40 is refresh. Then 60, etc.
        # In our measurement_step (0-indexed), refresh at 10, 30.
        # 정확하게는 stat의 cert_eval_count 증감으로 판단해야 하지만 근사:
        cur_steps = sc_dion.state[params[0]]['step_count']
        is_refresh.append(cur_steps % 20 == 1)  # just after refresh
    
    # ========== Analysis ==========
    times_muon_arr = np.array(times_muon)
    times_sc_arr = np.array(times_sc)
    
    log(f"\n  ━━━ Muon ━━━")
    log(f"    median:  {np.median(times_muon_arr):.1f} ms")
    log(f"    mean:    {np.mean(times_muon_arr):.1f} ms")
    log(f"    min:     {np.min(times_muon_arr):.1f} ms")
    log(f"    max:     {np.max(times_muon_arr):.1f} ms")
    log(f"    p5-p95:  [{np.percentile(times_muon_arr, 5):.1f}, "
        f"{np.percentile(times_muon_arr, 95):.1f}] ms")
    
    log(f"\n  ━━━ SC-Dion ━━━")
    log(f"    median:  {np.median(times_sc_arr):.1f} ms")
    log(f"    mean:    {np.mean(times_sc_arr):.1f} ms")
    log(f"    min:     {np.min(times_sc_arr):.1f} ms")
    log(f"    max:     {np.max(times_sc_arr):.1f} ms")
    log(f"    p5-p95:  [{np.percentile(times_sc_arr, 5):.1f}, "
        f"{np.percentile(times_sc_arr, 95):.1f}] ms")
    
    # Outlier-filtered analysis (drop top/bottom 5%)
    log(f"\n  ━━━ Outlier-trimmed comparison (5%-95%) ━━━")
    trim_lo, trim_hi = np.percentile(times_sc_arr, [5, 95])
    times_sc_trim = times_sc_arr[(times_sc_arr >= trim_lo) & (times_sc_arr <= trim_hi)]
    trim_lo_m, trim_hi_m = np.percentile(times_muon_arr, [5, 95])
    times_muon_trim = times_muon_arr[(times_muon_arr >= trim_lo_m) & (times_muon_arr <= trim_hi_m)]
    
    muon_typical = np.median(times_muon_trim)
    sc_typical = np.median(times_sc_trim)
    
    log(f"    Muon typical:    {muon_typical:.1f} ms")
    log(f"    SC-Dion typical: {sc_typical:.1f} ms")
    log(f"    Ratio:           {sc_typical / muon_typical:.2f}x")
    
    # SC-Dion fraction
    cs = sc_dion.get_cert_stats()
    log(f"\n  ━━━ SC-Dion stats ━━━")
    log(f"    Cert pass rate (random matrices): {cs['cert_pass_rate']:.2f}")
    log(f"    SC-Dion mode fraction: {cs['sc_dion_fraction']:.2f}")
    log(f"    Refresh count during measurement: {cs['cert_eval_count']}")
    
    # Final verdict
    log("")
    if sc_typical < muon_typical * 0.7:
        log(f"  🎯 STRONG WIN: SC-Dion runs at {sc_typical/muon_typical:.2f}x Muon time")
        log(f"     α=0.5 절감이 진단 cost를 압도함")
    elif sc_typical < muon_typical:
        log(f"  ✅ FASTER: SC-Dion at {sc_typical/muon_typical:.2f}x Muon time")
    elif sc_typical < muon_typical * 1.2:
        log(f"  ✅ COMPETITIVE: SC-Dion at {sc_typical/muon_typical:.2f}x Muon (within 20%)")
    elif sc_typical < muon_typical * 2.0:
        log(f"  ⚠️ SLOWER: SC-Dion at {sc_typical/muon_typical:.2f}x Muon (1.2-2x)")
    else:
        log(f"  ❌ MUCH SLOWER: SC-Dion at {sc_typical/muon_typical:.2f}x Muon (>2x)")


def convergence_test():
    """Toy 문제에서 SCDionGPU가 SCDion (CPU)과 같은 수렴 패턴을 보이는지."""
    log("\n" + "="*70)
    log("Test 4: Convergence on toy coverable problem")
    log("="*70)
    
    W_star = make_coverable_target(m=128, n=128, rank=4, support_size=12, seed=42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    W_star_dev = W_star.to(device)
    
    def run(opt_name, alpha=0.25, lr=0.3, max_steps=300, target=0.1):
        W = torch.zeros_like(W_star_dev, requires_grad=True)
        if opt_name == 'muon':
            opt = Muon([W], lr=lr)
        elif opt_name == 'sc_dion_cpu':
            opt = SCDion([W], lr=lr, alpha_u=alpha, alpha_d=1.0,
                        subspace_rank=4, cert_threshold=0.05, refresh_period=5)
        elif opt_name == 'sc_dion_gpu':
            opt = SCDionGPU([W], lr=lr, alpha_u=alpha, alpha_d=1.0,
                            subspace_rank=4, cert_threshold=0.05, refresh_period=5)
        
        init_err = float(torch.norm(W_star_dev).item())
        steps_to_target = None
        for step in range(max_steps):
            opt.zero_grad()
            loss = 0.5 * ((W - W_star_dev) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                rel = float(torch.norm(W - W_star_dev).item()) / init_err
            if steps_to_target is None and rel <= target:
                steps_to_target = step + 1
                break
        return steps_to_target, rel
    
    log(f"  {'Optimizer':<25} {'α':<6} {'Steps':<8} {'Final err':<10}")
    log("  " + "-"*55)
    
    s, e = run('muon')
    log(f"  {'muon':<25} {'-':<6} {str(s):<8} {e:<10.4f}")
    
    for alpha in [0.5, 0.25, 0.125]:
        s, e = run('sc_dion_cpu', alpha=alpha)
        log(f"  {'sc_dion_cpu':<25} {alpha:<6} {str(s):<8} {e:<10.4f}")
        s, e = run('sc_dion_gpu', alpha=alpha)
        log(f"  {'sc_dion_gpu':<25} {alpha:<6} {str(s):<8} {e:<10.4f}")
    
    log("\n  Expected: sc_dion_gpu와 sc_dion_cpu가 비슷한 step에 도달")
    log("            (정확히 같진 않지만 ±20% 이내)")


def main():
    log("\n" + "#"*70)
    log("# SC-Dion GPU Implementation Validation")
    log("#"*70)
    
    if torch.cuda.is_available():
        log(f"\nDevice: {torch.cuda.get_device_name(0)}")
        log(f"PyTorch: {torch.__version__}")
        log(f"CUDA:    {torch.version.cuda}")
    
    numerical_agreement_test()
    speed_benchmark()
    optimizer_step_benchmark()
    convergence_test()
    
    log("\n" + "#"*70)
    log("# Validation complete")
    log("#"*70)


if __name__ == '__main__':
    main()
