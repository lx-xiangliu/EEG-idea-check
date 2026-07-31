import torch

from src.attention import TRFPartialAttention


def test_gradients_flow_to_qkv_but_not_covariates() -> None:
    q = torch.randn(2, 10, 5, requires_grad=True)
    k = torch.randn(2, 10, 5, requires_grad=True)
    v = torch.randn(2, 10, 6, requires_grad=True)
    c = torch.randn(2, 10, 3, requires_grad=True)
    output, _ = TRFPartialAttention()(q, k, v, c, c)
    output.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert c.grad is None

