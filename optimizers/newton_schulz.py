"""
Newton-Schulz orthogonalization (Muon/Dion 공통 핵심).

성능 최적화:
1. bf16 native: A100 tensor core 활용. fp32 cast 제거.
2. coefficients를 buffer로: torch.compile recompile 방지.
3. spectral_normalize 분리: NS 핵심 polynomial과 분리하여 compile 효율 ↑

참고: Jordan et al. 2024 Muon은 bf16에서 정상 작동함이 검증됨.
"""
import torch


# Standard Muon coefficients (Jordan et al. 2024)
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315


def _ns_polynomial(X: torch.Tensor) -> torch.Tensor:
    """X <- a*X + (b*A + c*A^2) X  where A = X X^T.
    
    bf16 직접 연산 (A100 tensor core 활용).
    """
    A = X @ X.transpose(-2, -1)
    B = NS_B * A + NS_C * (A @ A)
    return NS_A * X + B @ X


def newton_schulz(G: torch.Tensor, num_steps: int = 5,
                  eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization. 입력 G의 polar factor U를 근사.
    
    Args:
        G: (..., m, n). bf16/fp16/fp32 모두 OK. bf16 권장 (A100 tensor core).
        num_steps: NS iteration (q in paper).
        eps: Frobenius normalization 안정성.
    
    Returns:
        U: orthogonalized, same shape and dtype as G.
    """
    # bf16/fp16은 그대로 유지 (cast 안 함 - tensor core 활용)
    X = G
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    
    # Pre-scale: spectral norm <= 1.
    # Frobenius normalize는 보수적인 estimate of spectral norm
    norm = X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X / norm
    
    # NS iterations
    for _ in range(num_steps):
        X = _ns_polynomial(X)
    
    if transposed:
        X = X.transpose(-2, -1)
    return X


def is_2d_param(p: torch.nn.Parameter) -> bool:
    """Muon/Dion은 2D 행렬 파라미터에만 적용. 1D/임베딩은 AdamW로."""
    return p.ndim == 2 and min(p.shape) >= 2
