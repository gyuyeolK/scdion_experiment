"""
Muon optimizer (Jordan et al. 2024) - baseline.

각 2D 파라미터의 모멘텀 전체에 Newton-Schulz 적용.
"""
import torch
from torch.optim.optimizer import Optimizer
from .newton_schulz import newton_schulz, is_2d_param


class Muon(Optimizer):
    """
    Standard Muon. 1D 파라미터는 별도 그룹에서 AdamW로 처리하는 것을 권장.
    """
    
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
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
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                if not is_2d_param(p):
                    raise ValueError(
                        f"Muon requires 2D matrix params, got shape {p.shape}. "
                        "Use a separate AdamW group for 1D / embedding params."
                    )
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                M = state['momentum_buffer']
                G = p.grad
                
                # EMA-style momentum (Muon convention: M <- momentum*M + G)
                M.mul_(momentum).add_(G)
                
                # Orthogonalize entire momentum matrix
                U = newton_schulz(M, num_steps=ns_steps)
                
                # Spectral-norm-aware step size (Muon scaling)
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(U, alpha=-lr * scale)
        
        return loss
