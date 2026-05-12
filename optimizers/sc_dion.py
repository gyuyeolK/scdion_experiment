"""
SC-Dion: Certified Selector-aware Dion (논문 Sec 4).

핵심 아이디어:
1. 모멘텀 S_t의 dominant rank-k left subspace U_{t,k}를 (sketched) SVD로 추정
2. 그 subspace를 잘 "cover"하는 행들을 greedy log-det로 선택
3. 매 스텝 증명서 ĉ_t = sqrt(ω̂_t)(1 - τ̂_t) - τ̂_t 를 평가
   - 통과: SC-Dion 모드 (alpha_u 비율 행만 NS) → 절감
   - 실패: full Muon으로 폴백 → 손해 없음
4. Update selector와 decay selector를 분리 (decay는 uniform 유지 → 추적 증명 보존)

증명서 통과 시: T_D/T_M = O(1) 보장 (Cor 4.3)
"""
import torch
from torch.optim.optimizer import Optimizer
from .newton_schulz import newton_schulz, is_2d_param


def _randomized_subspace(S: torch.Tensor, k: int, oversample: int = 5,
                         n_iter: int = 2) -> tuple[torch.Tensor, float]:
    """
    Randomized range finder (Halko et al. 2011) for dominant left subspace.
    
    Returns:
        U_k: (m, k) approximate dominant left singular vectors
        tau: dominant-tail ratio ||S - S_k||_* / ||S||_* (nuclear norm)
             증명서의 핵심 입력.
    """
    m, n = S.shape
    k = min(k, min(m, n) - 1)
    l = k + oversample
    
    # Subspace iteration: Q <- orth(S Omega), then Q <- orth(S S^T Q)
    Omega = torch.randn(n, l, device=S.device, dtype=S.dtype)
    Y = S @ Omega
    Q, _ = torch.linalg.qr(Y)
    for _ in range(n_iter):
        Z = S.t() @ Q
        Q_, _ = torch.linalg.qr(Z)
        Y = S @ Q_
        Q, _ = torch.linalg.qr(Y)
    
    # Project: B = Q^T S (l x n), then small SVD
    B = Q.t() @ S
    # Small SVD on (l, n) - cheap since l << min(m,n)
    Ub, sb, _ = torch.linalg.svd(B, full_matrices=False)
    U_k = (Q @ Ub)[:, :k]  # (m, k)
    sigma_k = sb[:k]
    
    # Nuclear-norm tail estimate.
    # 정확한 ||S||_* 는 비싸므로, 짧은 차원의 trace norm estimator 사용:
    # ||S||_* >= sum(sigma_k) (because singular values are nonneg).
    # tau의 보수적 추정: 1 - sum(sigma_k) / S_nuc_estimate.
    # Frobenius / nuclear 관계 ||A||_F <= ||A||_* <= sqrt(rank) ||A||_F 활용 시
    # 가장 안전한 보수적 추정:
    s_frob_sq = (S * S).sum().item()
    s_frob = max(s_frob_sq, 1e-20) ** 0.5
    top_nuc = sigma_k.sum().item()
    # ||S||_* 의 보수적 하한은 Frobenius norm. 상한은 sqrt(min(m,n)) * F.
    # τ는 잘려나간 부분의 비중이므로, 정확한 값보다는 잘 보정된 추정이면 됨.
    # 여기선 다음 휴리스틱: ||S||_* ≈ sqrt(effective_rank) * F where effective_rank
    # 를 sb의 분포로 추정.
    # 더 견고한 방법: 모든 sb를 본다.
    if sb.numel() > k:
        # 우리는 (l, n) 작은 행렬의 전체 특이값을 이미 계산했으므로
        full_sing_approx = sb  # 위에서 top-l 특이값까지 구함
        total_nuc_approx = full_sing_approx.sum().item()
        tau = max(0.0, 1.0 - top_nuc / max(total_nuc_approx, 1e-20))
    else:
        tau = 0.0
    
    return U_k, tau


def _greedy_logdet_select(U_k: torch.Tensor, num_select: int) -> torch.Tensor:
    """
    Greedy log-det maximization (DPP / volume-sampling style).
    
    U_k: (m, k) → 부분공간을 행 partial selection으로 cover.
    선택된 행들의 Gram matrix det을 최대화.
    
    Returns: (num_select,) long tensor of selected row indices.
    """
    m, k = U_k.shape
    if num_select >= m:
        return torch.arange(m, device=U_k.device)
    
    # 초기: 가장 큰 norm 행 선택
    norms_sq = (U_k * U_k).sum(dim=1)
    selected = [int(torch.argmax(norms_sq).item())]
    
    # Cholesky update 형식의 incremental log-det
    # G = U_S^T U_S (k x k). 새 행 추가 시 schur complement로 marginal gain 계산.
    eps = 1e-8
    G_inv = None  # (|S|<=k이면 미정의, k 이상부터 의미 있음)
    
    # 단순화: 매 스텝마다 small k x k Gram의 logdet을 직접 계산
    # k가 작으므로 (보통 k=8~32) O(num_select * m * k^2)로 충분히 빠름
    
    available = torch.ones(m, dtype=torch.bool, device=U_k.device)
    available[selected[0]] = False
    
    for _ in range(num_select - 1):
        # 현재 선택된 행렬
        U_S = U_k[selected, :]  # (|S|, k)
        # 후보 인덱스
        cand_idx = torch.where(available)[0]
        if cand_idx.numel() == 0:
            break
        U_cand = U_k[cand_idx, :]  # (|cand|, k)
        
        # G_S = U_S^T U_S + eps*I
        G_S = U_S.t() @ U_S + eps * torch.eye(k, device=U_k.device)
        # 각 후보 j에 대해 새 행을 더하면 logdet 증가량은
        # log(1 + u_j^T G_S_minus_eps^{-1} u_j) 형태이지만, eps regularization 포함이라
        # 직접 logdet(G_S + u_j u_j^T)를 계산하는 게 안정적.
        # marginal: logdet(G + u u^T) - logdet(G) = log(1 + u^T G^{-1} u)
        try:
            G_S_inv = torch.linalg.inv(G_S)
        except RuntimeError:
            G_S_inv = torch.linalg.pinv(G_S)
        
        # gain[j] = log(1 + u_j^T G_S_inv u_j)
        # u_j^T G_S_inv u_j  for all j: diag(U_cand @ G_S_inv @ U_cand^T)
        Gu = U_cand @ G_S_inv  # (|cand|, k)
        quad = (Gu * U_cand).sum(dim=1)  # (|cand|,)
        gain = torch.log1p(torch.clamp(quad, min=-0.999))
        
        best_local = int(torch.argmax(gain).item())
        best_global = int(cand_idx[best_local].item())
        selected.append(best_global)
        available[best_global] = False
    
    return torch.tensor(selected, device=U_k.device, dtype=torch.long)


def _certificate(U_k: torch.Tensor, selected_idx: torch.Tensor, tau: float
                 ) -> tuple[float, float]:
    """
    증명서: ĉ = sqrt(ω̂)(1 - τ̂) - τ̂
    
    ω̂ = λ_min(U_k[I,:]^T U_k[I,:])  ← 선택된 행이 부분공간을 얼마나 균일하게 덮나
    
    Returns: (cert, omega)
    """
    U_sel = U_k[selected_idx, :]
    k = U_sel.size(1)
    G = U_sel.t() @ U_sel + 1e-10 * torch.eye(k, device=U_sel.device)
    try:
        eigvals = torch.linalg.eigvalsh(G)
        omega = float(eigvals.min().item())
    except RuntimeError:
        omega = 0.0
    omega = max(0.0, omega)
    cert = (omega ** 0.5) * (1.0 - tau) - tau
    return cert, omega


class SCDion(Optimizer):
    """
    Certified Selector-aware Dion.
    
    매 K 스텝마다 dominant subspace 갱신. 매 스텝 증명서 평가:
    - 통과 → SC-Dion 모드 (alpha_u 행만 NS)
    - 실패 → full Muon 폴백
    
    decay는 uniform random (alpha_d) 분리 적용 → 분석 보존.
    
    Args:
        alpha_u: update fraction (NS 적용 행 비율, 작을수록 절감)
        alpha_d: decay fraction (1.0 권장 → 추가 noise 없음)
        subspace_rank: k (dominant subspace 추정 차원)
        cert_threshold: c_min (증명서 통과 기준치)
        refresh_period: K (subspace를 K 스텝마다 다시 계산)
    """
    
    def __init__(self, params, lr=1e-3, alpha_u=0.5, alpha_d=1.0, mu=0.95,
                 ns_steps=5, subspace_rank=8, cert_threshold=0.05,
                 refresh_period=10, weight_decay=0.0):
        if not 0 < alpha_u <= 1:
            raise ValueError(f"alpha_u must be in (0,1], got {alpha_u}")
        if not 0 < alpha_d <= 1:
            raise ValueError(f"alpha_d must be in (0,1], got {alpha_d}")
        defaults = dict(lr=lr, alpha_u=alpha_u, alpha_d=alpha_d, mu=mu,
                        ns_steps=ns_steps, subspace_rank=subspace_rank,
                        cert_threshold=cert_threshold,
                        refresh_period=refresh_period, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # 통계 추적 (논문의 falsifiable prediction 검증용)
        self.stats = {
            'cert_pass_count': 0,
            'cert_fail_count': 0,
            'cert_values': [],  # 마지막 N개만 보관
            'tau_values': [],
            'omega_values': [],
        }
        self._stats_buffer_max = 1000
    
    def _maybe_truncate_stats(self):
        for key in ['cert_values', 'tau_values', 'omega_values']:
            if len(self.stats[key]) > self._stats_buffer_max:
                self.stats[key] = self.stats[key][-self._stats_buffer_max:]
    
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
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                if not is_2d_param(p):
                    raise ValueError(f"SCDion requires 2D params, got {p.shape}")
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step_count'] = 0
                    state['cached_U_k'] = None
                    state['cached_tau'] = None
                
                M = state['momentum_buffer']
                G = p.grad
                S = M + G
                step_count = state['step_count']
                
                # Shorter-dimension orientation
                transposed = S.size(0) > S.size(1)
                if transposed:
                    S_view = S.t().contiguous()
                else:
                    S_view = S
                m = S_view.size(0)
                k_eff = min(k, m - 1, S_view.size(1) - 1)
                k_eff = max(1, k_eff)
                
                # === Subspace refresh (every K steps) ===
                if state['cached_U_k'] is None or step_count % K == 0:
                    U_k, tau = _randomized_subspace(S_view, k=k_eff)
                    state['cached_U_k'] = U_k
                    state['cached_tau'] = tau
                else:
                    U_k = state['cached_U_k']
                    tau = state['cached_tau']
                
                # === Selector: greedy log-det subspace cover ===
                num_select = max(1, int(round(alpha_u * m)))
                if num_select >= m:
                    # alpha_u = 1: equivalent to Muon, no selection needed
                    sel_idx = torch.arange(m, device=S_view.device)
                    cert = 1.0  # trivially passes
                    omega = 1.0
                    use_full = True
                else:
                    sel_idx = _greedy_logdet_select(U_k, num_select)
                    cert, omega = _certificate(U_k, sel_idx, tau)
                    use_full = cert < c_min
                
                # 통계 기록
                self.stats['cert_values'].append(float(cert))
                self.stats['tau_values'].append(float(tau))
                self.stats['omega_values'].append(float(omega))
                if use_full:
                    self.stats['cert_fail_count'] += 1
                else:
                    self.stats['cert_pass_count'] += 1
                
                # === Update: NS on selected block (or full if cert failed) ===
                if use_full:
                    # Fallback to full Muon
                    U_update = newton_schulz(S_view, num_steps=ns_steps)
                    O_view = U_update
                else:
                    A = S_view[sel_idx, :]
                    U_block = newton_schulz(A, num_steps=ns_steps)
                    O_view = torch.zeros_like(S_view)
                    O_view[sel_idx, :] = U_block
                
                if transposed:
                    O = O_view.t().contiguous()
                else:
                    O = O_view
                
                # Apply step
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)
                
                # === Decay: uniform random support (independent of update selector) ===
                # alpha_d = 1: 모든 행 decay (cleanest theoretical variant)
                if alpha_d >= 0.999:
                    M_new = mu * S_view  # standard EMA on entire matrix
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
        
        self._maybe_truncate_stats()
        return loss
    
    def get_cert_stats(self) -> dict:
        """현재까지의 증명서 통과율 등 진단 통계."""
        total = self.stats['cert_pass_count'] + self.stats['cert_fail_count']
        pass_rate = self.stats['cert_pass_count'] / max(1, total)
        recent_cert = self.stats['cert_values'][-200:] if self.stats['cert_values'] else [0]
        recent_tau = self.stats['tau_values'][-200:] if self.stats['tau_values'] else [0]
        recent_omega = self.stats['omega_values'][-200:] if self.stats['omega_values'] else [0]
        return {
            'cert_pass_rate': pass_rate,
            'cert_pass_count': self.stats['cert_pass_count'],
            'cert_fail_count': self.stats['cert_fail_count'],
            'recent_cert_mean': sum(recent_cert) / len(recent_cert),
            'recent_tau_mean': sum(recent_tau) / len(recent_tau),
            'recent_omega_mean': sum(recent_omega) / len(recent_omega),
        }
