import torch

from src.attention import TRFPartialAttention


def test_partial_attention_outputs_and_diagnostics() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 12, 8)
    k = torch.randn(2, 12, 8)
    v = torch.randn(2, 12, 6)
    c = torch.randn(2, 12, 3)
    output, diagnostics = TRFPartialAttention()(q, k, v, c, c, return_attention=True)
    assert output.shape == (2, 12, 6)
    assert diagnostics["attention"].shape == (2, 12, 12)
    assert torch.allclose(diagnostics["attention"].sum(-1), torch.ones(2, 12), atol=1e-5)

