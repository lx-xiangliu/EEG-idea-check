import torch

from src.trf import LaggedDesignBuilder


def test_lag_sign_and_mask() -> None:
    x = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    design, mask = LaggedDesignBuilder()(x, torch.tensor([-1, 0, 2]))
    assert design.shape == (1, 5, 3)
    assert torch.equal(design[0, :, 1], x[0, :, 0])
    assert design[0, 3, 2] == x[0, 1, 0]
    assert mask.tolist() == [[False, False, True, True, False]]

