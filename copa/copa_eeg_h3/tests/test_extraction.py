import numpy as np
import torch

from src.extraction import _attention_summary, _token_summary


def test_bounded_summaries_are_finite():
    tokens = torch.randn(3, 11, 8)
    summary = _token_summary(tokens)
    assert summary.shape == (3, 5 * 8 + 2)
    assert np.isfinite(summary).all()
    logits = torch.randn(3, 2, 11, 11)
    probs = logits.softmax(-1)
    global_summary, per_head, stats = _attention_summary(probs, True)
    assert global_summary.shape == (3, 14)
    assert len(per_head) == 2
    np.testing.assert_allclose(
        stats["diagonal_mass"] + stats["off_diagonal_mass"], 1.0, atol=1e-6
    )

