"""
Dion2 (Ahn et al. 2025) with uniform-random row selection.

핵심 차이 vs Muon:
1. 매 스텝 alpha 비율의 행만 선택해 그 부분만 NS
2. Selective decay: 선택된 행에 대해서만 momentum 감쇠
3. Shorter-dimension 구현: m > n이면 transpose해서 더 작은 차원으로 선택
"""
import torch
from torch.optim.optimizer import Optimizer
from .newton_schulz import newton_schulz, is_2d_param


class Dion2Uniform(Optimizer):
    """
    Uniform-random Dion2.
    
    Args:
        alpha: row selection fraction in (0, 1]. alpha=1 reduces to Muon.
        mu: momentum decay (논문 표기 따라 mu, 1-mu 가 EMA에서의 가중치).
    """
    
    def __init__(self, params, lr=1e-3, alpha=0.5, mu=0.95, ns_steps=5,
                 weight_decay=0.0):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        defaults = dict(lr=lr, alpha=alpha, mu=mu, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            mu = group['mu']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                if not is_2d_param(p):
                    raise ValueError(f"Dion2 requires 2D params, got {p.shape}")
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                # Decide orientation: shorter-dimension along which we sample rows.
                # 논문 표기: r = min(m,n), s = max(m,n). 행 방향으로 선택하므로
                # m <= n이 되도록 view.
                M = state['momentum_buffer']
                G = p.grad
                
                # S_t = M_{t-1} + G_t (Algorithm 1, Line 4)
                S = M + G
                
                # m을 짧은 쪽으로
                transposed = S.size(0) > S.size(1)
                if transposed:
                    S_view = S.t()
                else:
                    S_view = S
                m = S_view.size(0)
                k = max(1, int(round(alpha * m)))
                
                # Uniform random row selection (Algorithm 1, Line 5)
                idx = torch.randperm(m, device=S.device)[:k]
                
                # Selected sub-block (Line 6)
                A = S_view[idx, :]
                
                # Newton-Schulz on the small block only - 핵심 절감
                U_block = newton_schulz(A, num_steps=ns_steps)
                
                # Embed back: O_t (Line 6)
                O_view = torch.zeros_like(S_view)
                O_view[idx, :] = U_block
                if transposed:
                    O = O_view.t()
                else:
                    O = O_view
                
                # Apply update (Line 7, first half)
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)
                
                # Selective decay (Line 7, second half):
                #   M_t = S_t - (1 - mu) P_t S_t
                # → 선택된 행만 (1-mu) S로 곱해서 EMA, 비선택 행은 S 그대로 유지
                # 즉 비선택 행 = M_{t-1} + G_t 그대로 누적, 선택 행 = mu*(M_{t-1}+G_t)
                # 이는 Muon의 EMA를 random subset에 대해서만 적용하는 형태.
                M_new = S_view.clone()
                M_new[idx, :] = mu * S_view[idx, :]
                if transposed:
                    state['momentum_buffer'] = M_new.t().contiguous()
                else:
                    state['momentum_buffer'] = M_new
        
        return loss
