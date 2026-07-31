import torch

from src.attention import TRFPartialAttention


def test_query_and_key_padding_never_contribute() -> None:
    q = torch.randn(1, 6, 4)
    k = torch.randn(1, 7, 4)
    v = torch.randn(1, 7, 3)
    qc = torch.randn(1, 6, 2)
    kc = torch.randn(1, 7, 2)
    qm = torch.tensor([[True, True, True, True, False, False]])
    km = torch.tensor([[True, True, True, True, True, False, False]])
    output, info = TRFPartialAttention()(q, k, v, qc, kc, qm, km, True)
    assert torch.equal(output[:, 4:], torch.zeros_like(output[:, 4:]))
    assert torch.equal(info["attention"][:, :, 5:], torch.zeros_like(info["attention"][:, :, 5:]))

