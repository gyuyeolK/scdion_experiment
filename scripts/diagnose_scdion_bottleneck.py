"""
SC-Dion refresh bottleneck 진단.

실제 SmolLM2-360M의 파라미터 shape을 가지고 refresh 한 번이 얼마나 걸리는지,
어느 파라미터(어느 shape)가 가장 느린지 측정.

사용:
    python scripts/diagnose_scdion_bottleneck.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from optimizers.sc_dion_gpu import (
    _randomized_subspace_gpu, _greedy_logdet_select_gpu, _certificate_gpu
)


def log(s):
    print(s, flush=True)


def time_one_param(shape, k=8, alpha=0.5, n_warmup=3, n_iter=5):
    """한 shape에 대해 refresh time 측정."""
    m, n = shape
    # Shorter-dimension orient
    transposed = m > n
    if transposed:
        m, n = n, m
    
    num_select = max(1, int(round(alpha * m)))
    
    S = torch.randn(m, n, device='cuda', dtype=torch.bfloat16) * 0.01
    
    # Warmup
    for _ in range(n_warmup):
        U_k, tau = _randomized_subspace_gpu(S, k=k)
        sel = _greedy_logdet_select_gpu(U_k, num_select)
        cert, omega = _certificate_gpu(U_k, sel, tau)
    torch.cuda.synchronize()
    
    # Time breakdown
    t_sub = []
    t_sel = []
    t_cert = []
    
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        U_k, tau = _randomized_subspace_gpu(S, k=k)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sel = _greedy_logdet_select_gpu(U_k, num_select)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cert, omega = _certificate_gpu(U_k, sel, tau)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        t_sub.append((t1 - t0) * 1000)
        t_sel.append((t2 - t1) * 1000)
        t_cert.append((t3 - t2) * 1000)
    
    return {
        'shape': (m, n),
        'num_select': num_select,
        'sub_ms': sum(t_sub) / len(t_sub),
        'sel_ms': sum(t_sel) / len(t_sel),
        'cert_ms': sum(t_cert) / len(t_cert),
        'total_ms': (sum(t_sub) + sum(t_sel) + sum(t_cert)) / len(t_sub),
    }


def main():
    if not torch.cuda.is_available():
        log("CUDA required")
        return
    
    log(f"Device: {torch.cuda.get_device_name(0)}")
    
    # SmolLM2-360M parameter shapes (from inspection)
    # 30 layers × ~5-7 matrix params + embeddings
    # 주요 shape들 (m, n 그대로):
    shapes = [
        (49152, 960, "lm_head"),     # vocab × hidden
        (960, 2560, "ffn_gate"),     # hidden × ffn (×30 layers)
        (960, 2560, "ffn_up"),       # 
        (2560, 960, "ffn_down"),     # 
        (960, 960, "q_proj"),        # 
        (960, 320, "k_proj"),        # ← group attention
        (960, 320, "v_proj"),        # 
        (960, 960, "o_proj"),        # 
    ]
    
    log(f"\n{'Shape (orient)':<25} {'num_sel':<10} {'sub ms':<10} {'sel ms':<10} {'cert ms':<10} {'TOTAL':<10}")
    log("-" * 80)
    
    total_per_step = 0
    layer_counts = {  # SmolLM2-360M에 layer당 몇 개의 weight가 있는가
        'lm_head': 1,
        'ffn_gate': 30, 'ffn_up': 30, 'ffn_down': 30,
        'q_proj': 30, 'k_proj': 30, 'v_proj': 30, 'o_proj': 30,
    }
    
    grand_total_ms = 0
    
    for shape_tuple in shapes:
        m, n, name = shape_tuple
        r = time_one_param((m, n), k=8, alpha=0.5)
        count = layer_counts.get(name, 1)
        layer_total = r['total_ms'] * count
        grand_total_ms += layer_total
        
        orient_m, orient_n = r['shape']
        log(f"{name:<15} ({orient_m}x{orient_n}) ×{count:<3} "
            f"{r['num_select']:<10} "
            f"{r['sub_ms']:<10.2f} {r['sel_ms']:<10.2f} {r['cert_ms']:<10.2f} "
            f"{r['total_ms']:<10.2f} → ×{count} = {layer_total:.0f} ms")
    
    log(f"\n{'='*80}")
    log(f"TOTAL refresh time (one full pass over all params): {grand_total_ms:.0f} ms")
    log(f"   Expected: ~17000 ms (observed)")
    log(f"   Discrepancy: {17000/max(grand_total_ms,1):.1f}x")
    log(f"{'='*80}")
    
    # 가장 큰 문제 찾기
    log(f"\nLargest contributor: ", end="")
    biggest = max(shapes, key=lambda s: time_one_param((s[0], s[1]))['total_ms'] *
                                          layer_counts.get(s[2], 1))
    r = time_one_param((biggest[0], biggest[1]))
    log(f"{biggest[2]} ({r['shape']}, num_sel={r['num_select']}): "
        f"{r['total_ms']:.0f} ms × {layer_counts[biggest[2]]} = "
        f"{r['total_ms']*layer_counts[biggest[2]]:.0f} ms")
    
    # 더 자세한 진단: greedy_select에 num_select 의존성
    log("\n=== greedy_select scaling with num_select ===")
    log("(이것이 핵심 병목 - num_select가 클수록 loop가 길어짐)")
    log(f"\n{'shape':<20} {'num_select':<12} {'sel_ms':<10}")
    log("-" * 50)
    
    for m in [1024, 2048, 4096, 8192]:
        for alpha in [0.125, 0.25, 0.5]:
            num_sel = int(alpha * m)
            S = torch.randn(m, 4096, device='cuda', dtype=torch.bfloat16) * 0.01
            # warmup
            for _ in range(2):
                U_k, _ = _randomized_subspace_gpu(S, k=8)
                _greedy_logdet_select_gpu(U_k, num_sel)
            torch.cuda.synchronize()
            
            U_k, _ = _randomized_subspace_gpu(S, k=8)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _greedy_logdet_select_gpu(U_k, num_sel)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / 3 * 1000
            log(f"  m={m:<5} n=4096   num_sel={num_sel:<8} (α={alpha}): {t:.1f} ms")


if __name__ == '__main__':
    main()
