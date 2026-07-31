import pytest
import torch

from src.trf import ResidualMaker


@pytest.mark.parametrize("method,ridge", [("qr", 0.0), ("ridge", 1e-6)])
def test_removes_covariate_and_handles_rank_deficiency(method: str, ridge: float) -> None:
    torch.manual_seed(0)
    c = torch.randn(2, 20, 2)
    c = torch.cat([c, c[..., :1]], dim=-1)
    weight = torch.randn(2, 3, 4)
    x = c @ weight + 0.01 * torch.randn(2, 20, 4)
    residual = ResidualMaker(ridge=ridge, method=method)(x, c)
    assert residual.shape == x.shape
    assert residual.square().mean() < x.square().mean() * 0.01


def test_masked_padding_is_zero() -> None:
    x = torch.randn(1, 8, 3)
    c = torch.randn(1, 8, 2)
    mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    residual = ResidualMaker()(x, c, mask)
    assert torch.equal(residual[:, 4:], torch.zeros_like(residual[:, 4:]))

