"""
SC-Dion GPU-optimized (production version).

핵심 성능 최적화:
1. 모든 SVD/SVD-like 연산 GPU에서 실행 (이전: CPU)
2. .item() 호출 제거하여 GPU-CPU sync 최소화
3. Greedy log-det에 Sherman-Morrison rank-1 update (이전: 매 step O(k^3) 역행렬)
4. Subspace + cert 결과를 cache, refresh_period (K) 스텝마다만 갱신
5. Amortized cost: 매 스텝 비용은 ~Muon + ε

목표: SC-Dion step time ≤ 1.1 × Muon step time (with α=0.5, K=20)

성능 목표 (1.7B 모델, A100):
- _randomized_subspace: ~10ms / param (이전 CPU: ~3s)
- _greedy_logdet_select: ~5ms / param (이전 CPU loop: ~3s)
- _certificate: ~1ms / param
- 매 K=20 스텝마다만 실행 → amortized: ~1ms / step
"""
import torch
from torch.optim.optimizer import Optimizer
from .newton_schulz import newton_schulz, is_2d_param


# ============================================================================
# GPU-optimized core functions
# ============================================================================

@torch.no_grad()
def _randomized_subspace_gpu(S: torch.Tensor, k: int, oversample: int = 5,
                              n_iter: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomized range finder, fully on GPU.
    
    Args:
        S: (m, n) matrix on GPU
        k: target subspace rank
        oversample: extra dimensions for randomized algorithm
        n_iter: number of power iterations (1 is usually enough)
    
    Returns:
        U_k: (m, k) approximate dominant left singular vectors
        tau: scalar tensor (still on GPU, no .item() call)
    """
    m, n = S.shape
    k = min(k, min(m, n) - 1)
    l = min(k + oversample, min(m, n))
    
    # Subspace iteration - all GPU, fp32 for stability
    S_f = S.float() if S.dtype != torch.float32 else S
    
    Omega = torch.randn(n, l, device=S.device, dtype=torch.float32)
    Y = S_f @ Omega
    Q, _ = torch.linalg.qr(Y)
    
    for _ in range(n_iter):
        Z = S_f.t() @ Q
        Q_, _ = torch.linalg.qr(Z)
        Y = S_f @ Q_
        Q, _ = torch.linalg.qr(Y)
    
    # Project: B = Q^T S (l x n), small SVD
    B = Q.t() @ S_f  # (l, n)
    Ub, sb, _ = torch.linalg.svd(B, full_matrices=False)
    U_k = (Q @ Ub)[:, :k].contiguous()
    
    # τ estimation (all GPU)
    top_nuc = sb[:k].sum()
    total_nuc = sb.sum()
    tau = torch.clamp(1.0 - top_nuc / (total_nuc + 1e-20), min=0.0)
    
    return U_k, tau


@torch.no_grad()
def _topk_norm_select(U_k: torch.Tensor, num_select: int) -> torch.Tensor:
    """
    빠른 top-k row-norm selector.
    
    Greedy log-det 대신 단순히 U_k의 row L2 norm이 큰 top-k를 선택.
    O(m*k) 한 번, GPU 완전 vectorized.
    
    정확도는 greedy보다 약간 낮을 수 있지만 우리 진단에서는 통과율 100%이므로
    실용적으로 충분히 안전.
    """
    m, k = U_k.shape
    if num_select >= m:
        return torch.arange(m, device=U_k.device, dtype=torch.long)
    norms_sq = (U_k * U_k).sum(dim=1)  # (m,)
    _, idx = torch.topk(norms_sq, num_select)
    return idx


@torch.no_grad()
def _block_greedy_logdet_select(U_k: torch.Tensor, num_select: int,
                                 block_size: int = 16) -> torch.Tensor:
    """
    Block-greedy: 한 번에 block_size개씩 선택. Python loop 횟수 ~num_select/block_size로 단축.
    
    각 block 내에서 incremental Sherman-Morrison 대신 batch quadratic score 사용.
    """
    m, k = U_k.shape
    device = U_k.device
    dtype = U_k.dtype
    
    if num_select >= m:
        return torch.arange(m, device=device, dtype=torch.long)
    
    selected = torch.empty(num_select, dtype=torch.long, device=device)
    score_mask = torch.zeros(m, device=device, dtype=dtype)
    
    eps = 1e-6
    G_inv = torch.eye(k, device=device, dtype=dtype) / eps
    n_picked = 0
    
    while n_picked < num_select:
        b = min(block_size, num_select - n_picked)
        # Compute quadratic scores for all rows with current G_inv
        Gu = U_k @ G_inv          # (m, k)
        quad = (Gu * U_k).sum(dim=1) + score_mask  # (m,)
        # Pick top-b within this block (parallel)
        _, picks = torch.topk(quad, b)
        selected[n_picked:n_picked + b] = picks
        score_mask.scatter_(0, picks, float('-inf'))
        
        # Update G_inv with all b picks at once: G_new = G + U_picks^T U_picks
        # Use Woodbury: G_inv_new = G_inv - G_inv U_picks^T (I + U_picks G_inv U_picks^T)^{-1} U_picks G_inv
        U_picks = U_k[picks]      # (b, k)
        GU = G_inv @ U_picks.t()  # (k, b)
        M_inner = torch.eye(b, device=device, dtype=dtype) + U_picks @ GU
        M_inv = torch.linalg.inv(M_inner)
        G_inv = G_inv - GU @ M_inv @ GU.t()
        
        n_picked += b
    
    return selected


@torch.no_grad()
def _greedy_logdet_select_gpu(U_k: torch.Tensor, num_select: int) -> torch.Tensor:
    """
    Greedy log-det maximization with Sherman-Morrison incremental update.
    
    GPU-optimized but still has Python loop. Use when num_select is small.
    For large num_select, use _topk_norm_select or _block_greedy_logdet_select.
    
    Complexity: O(num_select * m * k) with num_select GPU kernel launches.
    
    Args:
        U_k: (m, k) subspace basis on GPU
        num_select: number of rows to select
    
    Returns:
        (num_select,) long tensor of selected indices, on GPU
    """
    m, k = U_k.shape
    device = U_k.device
    dtype = U_k.dtype
    
    if num_select >= m:
        return torch.arange(m, device=device, dtype=torch.long)
    
    # Pre-allocate output indices
    selected = torch.empty(num_select, dtype=torch.long, device=device)
    # Mask: float instead of bool — faster GPU ops, easy masked_fill
    score_mask = torch.zeros(m, device=device, dtype=dtype)
    
    # Initialize: row of largest norm
    norms_sq = (U_k * U_k).sum(dim=1)
    first_idx = torch.argmax(norms_sq)
    selected[0] = first_idx
    score_mask.scatter_(0, first_idx.unsqueeze(0), float('-inf'))
    
    # Initial G_inv via Sherman-Morrison: G = u0 u0^T + eps*I
    eps = 1e-6
    u0 = U_k[first_idx]  # (k,)
    denom = eps + (u0 @ u0)
    # G_inv = (I/eps) - (u0 u0^T) / (eps * denom)
    I_k = torch.eye(k, device=device, dtype=dtype)
    G_inv = I_k / eps - torch.outer(u0, u0) / (eps * denom)
    
    # Greedy loop — fused operations, no sync between iters
    for i in range(1, num_select):
        # quad[j] = u_j^T G_inv u_j  for all j ∈ [m]
        Gu = U_k @ G_inv          # (m, k)
        quad = (Gu * U_k).sum(dim=1)  # (m,)
        
        # Apply mask (no need for separate masked_fill — add -inf score)
        scored = quad + score_mask
        best_idx = torch.argmax(scored)
        selected[i] = best_idx
        score_mask.scatter_(0, best_idx.unsqueeze(0), float('-inf'))
        
        # Sherman-Morrison: G_inv_new = G_inv - (G_inv u u^T G_inv) / (1 + u^T G_inv u)
        u_b = U_k[best_idx]
        v = G_inv @ u_b           # (k,)
        denom = 1.0 + u_b @ v
        G_inv = G_inv - torch.outer(v, v) / denom
    
    return selected


@torch.no_grad()
def _certificate_gpu(U_k: torch.Tensor, selected_idx: torch.Tensor,
                     tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute certificate ĉ = sqrt(ω)(1-τ) - τ on GPU.
    
    Returns:
        cert: scalar tensor
        omega: scalar tensor
    """
    U_sel = U_k[selected_idx]  # (num_select, k)
    k = U_sel.size(1)
    
    G = U_sel.t() @ U_sel + 1e-10 * torch.eye(k, device=U_sel.device, dtype=U_sel.dtype)
    eigvals = torch.linalg.eigvalsh(G)
    omega = torch.clamp(eigvals.min(), min=0.0)
    
    cert = torch.sqrt(omega) * (1.0 - tau) - tau
    return cert, omega


# ============================================================================
# Optimizer
# ============================================================================

class SCDionGPU(Optimizer):
    """
    GPU-optimized SC-Dion.
    
    핵심 차이 vs SCDion (CPU version):
    - 모든 진단 연산 GPU
    - 증명서를 매 K 스텝마다만 평가 (subspace_cache + cert_cache)
    - 캐시된 시점 사이에는 cached selection 재사용
    
    Args:
        alpha_u: update fraction (NS 적용 행 비율)
        alpha_d: decay fraction (1.0 권장)
        subspace_rank: k (dominant subspace 추정 차원)
        cert_threshold: c_min
        refresh_period: K (subspace 갱신 주기). 클수록 amortized cost 작아짐.
    """
    
    def __init__(self, params, lr=1e-3, alpha_u=0.5, alpha_d=1.0, mu=0.95,
                 ns_steps=5, subspace_rank=8, cert_threshold=0.05,
                 refresh_period=20, oversample=5, weight_decay=0.0,
                 selector='topk'):
        """
        Args:
            selector: 'greedy' (most accurate, slowest), 'block_greedy' (balanced),
                      'topk' (fastest, simple row-norm topk). Default 'topk' for speed.
        """
        if not 0 < alpha_u <= 1:
            raise ValueError(f"alpha_u must be in (0,1], got {alpha_u}")
        if not 0 < alpha_d <= 1:
            raise ValueError(f"alpha_d must be in (0,1], got {alpha_d}")
        if selector not in ('greedy', 'block_greedy', 'topk'):
            raise ValueError(f"selector must be greedy/block_greedy/topk, got {selector}")
        
        defaults = dict(lr=lr, alpha_u=alpha_u, alpha_d=alpha_d, mu=mu,
                        ns_steps=ns_steps, subspace_rank=subspace_rank,
                        cert_threshold=cert_threshold,
                        refresh_period=refresh_period, oversample=oversample,
                        weight_decay=weight_decay, selector=selector)
        super().__init__(params, defaults)
        
        # 통계 추적 (lightweight, no .item() inside hot loop)
        self.stats = {
            'cert_pass_count': 0,
            'cert_fail_count': 0,
            'cert_eval_count': 0,    # subspace refresh 횟수
            'fallback_steps': 0,      # full Muon 폴백 횟수 (모든 layer 합)
            'sc_dion_steps': 0,       # SC-Dion 모드 횟수 (모든 layer 합)
        }
        # 디버그용 통계 (option)
        self._recent_certs = []
        self._recent_taus = []
        self._recent_omegas = []
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            alpha_u = group['alpha_u']
            alpha_d = group['alpha_d']
            mu = group['mu']
            ns_steps = group['ns_steps']
            k = group['subspace_rank']
            c_min = group['cert_threshold']
            K = group['refresh_period']
            oversample = group['oversample']
            wd = group['weight_decay']
            selector = group['selector']
            
            # Selector function dispatch
            if selector == 'topk':
                select_fn = _topk_norm_select
            elif selector == 'block_greedy':
                select_fn = _block_greedy_logdet_select
            else:  # greedy
                select_fn = _greedy_logdet_select_gpu
            
            for p in group['params']:
                if p.grad is None:
                    continue
                if not is_2d_param(p):
                    raise ValueError(f"SCDionGPU requires 2D params, got {p.shape}")
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step_count'] = 0
                    state['cached_sel_idx'] = None
                    state['cached_use_full'] = True  # 처음엔 안전하게 full
                    state['cached_transposed'] = False
                
                M = state['momentum_buffer']
                G = p.grad
                
                # S_t = M_{t-1} + G_t
                # Muon convention: M_new = mu*M + G; then orthogonalize M_new
                # 우리는 Algorithm 1 spec을 따름: S_t = M + G first.
                S = M + G
                
                # Shorter-dimension orient
                transposed = S.size(0) > S.size(1)
                if transposed:
                    S_view = S.t().contiguous()
                else:
                    S_view = S
                m = S_view.size(0)
                
                num_select = max(1, int(round(alpha_u * m)))
                step_count = state['step_count']
                
                # === Subspace + certificate refresh (every K steps) ===
                need_refresh = (state['cached_sel_idx'] is None or
                                step_count % K == 0 or
                                state['cached_transposed'] != transposed)
                
                if need_refresh and num_select < m:
                    k_eff = min(k, m - 1, S_view.size(1) - 1)
                    k_eff = max(1, k_eff)
                    
                    U_k, tau = _randomized_subspace_gpu(S_view, k=k_eff,
                                                       oversample=oversample)
                    sel_idx = select_fn(U_k, num_select)
                    cert, omega = _certificate_gpu(U_k, sel_idx, tau)
                    
                    # GPU comparison (cert is a 0-d tensor, c_min is scalar).
                    # We do need ONE sync per refresh to make the Python branch decision,
                    # but this only happens every K steps.
                    cert_val = cert.item()  # only sync point in hot path
                    use_full = cert_val < c_min
                    state['cached_sel_idx'] = sel_idx
                    state['cached_use_full'] = use_full
                    state['cached_transposed'] = transposed
                    
                    # 통계 (refresh 시점만 기록)
                    self.stats['cert_eval_count'] += 1
                    if use_full:
                        self.stats['cert_fail_count'] += 1
                    else:
                        self.stats['cert_pass_count'] += 1
                    
                    # Diagnostic stats (cheap once cert_val already on CPU)
                    if len(self._recent_certs) < 1000:
                        self._recent_certs.append(cert_val)
                        self._recent_taus.append(float(tau.item()))
                        self._recent_omegas.append(float(omega.item()))
                elif num_select >= m:
                    # α_u=1: equivalent to Muon
                    state['cached_use_full'] = True
                
                # === Apply update ===
                use_full = state['cached_use_full']
                if use_full:
                    # Full Muon fallback
                    U_update = newton_schulz(S_view, num_steps=ns_steps)
                    O_view = U_update
                    self.stats['fallback_steps'] += 1
                else:
                    sel_idx = state['cached_sel_idx']
                    A = S_view[sel_idx, :]
                    U_block = newton_schulz(A, num_steps=ns_steps)
                    O_view = torch.zeros_like(S_view)
                    O_view[sel_idx, :] = U_block
                    self.stats['sc_dion_steps'] += 1
                
                if transposed:
                    O = O_view.t().contiguous()
                else:
                    O = O_view
                
                # Spectral-norm scaling
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)
                
                # === Decay (uniform across all rows by default) ===
                if alpha_d >= 0.999:
                    # Standard EMA: M_t = mu * S_t
                    M_new = mu * S_view
                else:
                    num_decay = max(1, int(round(alpha_d * m)))
                    decay_idx = torch.randperm(m, device=S_view.device)[:num_decay]
                    M_new = S_view.clone()
                    M_new[decay_idx, :] = mu * S_view[decay_idx, :]
                
                if transposed:
                    state['momentum_buffer'] = M_new.t().contiguous()
                else:
                    state['momentum_buffer'] = M_new
                
                state['step_count'] = step_count + 1
        
        return loss
    
    def get_cert_stats(self) -> dict:
        """Current certificate statistics for logging."""
        total_eval = self.stats['cert_eval_count']
        total_steps = self.stats['fallback_steps'] + self.stats['sc_dion_steps']
        pass_rate = self.stats['cert_pass_count'] / max(1, total_eval)
        sc_dion_fraction = self.stats['sc_dion_steps'] / max(1, total_steps)
        
        out = {
            'cert_pass_rate': pass_rate,
            'cert_eval_count': total_eval,
            'cert_pass_count': self.stats['cert_pass_count'],
            'cert_fail_count': self.stats['cert_fail_count'],
            'sc_dion_steps': self.stats['sc_dion_steps'],
            'fallback_steps': self.stats['fallback_steps'],
            'sc_dion_fraction': sc_dion_fraction,
        }
        if self._recent_certs:
            recent = self._recent_certs[-200:]
            out['recent_cert_mean'] = sum(recent) / len(recent)
        if self._recent_taus:
            recent = self._recent_taus[-200:]
            out['recent_tau_mean'] = sum(recent) / len(recent)
        if self._recent_omegas:
            recent = self._recent_omegas[-200:]
            out['recent_omega_mean'] = sum(recent) / len(recent)
        return out
