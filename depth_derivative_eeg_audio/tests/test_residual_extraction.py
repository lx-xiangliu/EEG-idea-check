import pytest
import torch

from src.models import DepthDerivativeExtractor


def test_raw_residual_is_adjacent_difference() -> None:
    states = [torch.zeros(2, 3, 4), torch.ones(2, 3, 4), torch.full((2, 3, 4), 3.0)]
    residuals = DepthDerivativeExtractor()(states, normalize=False)
    assert torch.equal(residuals[0], torch.ones_like(residuals[0]))
    assert torch.equal(residuals[1], torch.full_like(residuals[1], 2.0))


def test_shape_mismatch_is_not_silently_projected() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        DepthDerivativeExtractor()([torch.zeros(2, 3, 4), torch.zeros(2, 2, 4)])
