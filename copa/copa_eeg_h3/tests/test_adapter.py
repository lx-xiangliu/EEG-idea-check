import torch

from src.model import AttentionHookRecorder, TinyEEGTransformer


def test_adapter_outputs_and_forward_consistency():
    torch.manual_seed(4)
    model = TinyEEGTransformer(8, 500, patch_samples=25, dim=16, heads=4, layers=2).eval()
    x = torch.randn(3, 8, 500)
    with torch.inference_mode():
        direct = model(x)
        captured = model.forward_with_internals(x)
    torch.testing.assert_close(direct, captured["pooled_representation"], atol=1e-7, rtol=1e-6)
    assert captured["input_embedding"].shape == (3, 21, 16)
    assert captured["final_tokens"].shape == (3, 21, 16)
    assert len(captured["layers"]) == 2
    required = {
        "q", "k", "v", "attention_logits", "attention_probs",
        "attention_output", "residual_before_attention",
        "residual_after_attention", "mlp_output",
    }
    assert required <= set(captured["layers"][0])


def test_qkv_are_the_exact_forward_projections():
    torch.manual_seed(9)
    model = TinyEEGTransformer(8, 500, patch_samples=25, dim=16, heads=4, layers=1).eval()
    x = torch.randn(2, 8, 500)
    with torch.inference_mode():
        captured = model.forward_with_internals(x)
        normalized = model.blocks[0].norm1(captured["input_embedding"])
        manual = model.blocks[0].attn.qkv(normalized).reshape(2, 21, 3, 4, 4)
        manual_q, manual_k, manual_v = manual.permute(2, 0, 3, 1, 4).unbind(0)
    for name, expected in (("q", manual_q), ("k", manual_k), ("v", manual_v)):
        torch.testing.assert_close(captured["layers"][0][name], expected, atol=1e-7, rtol=1e-6)


def test_forward_hooks_are_non_mutating_and_capture_each_layer():
    torch.manual_seed(12)
    model = TinyEEGTransformer(8, 500, patch_samples=25, dim=16, heads=4, layers=2).eval()
    x = torch.randn(2, 8, 500)
    with torch.inference_mode():
        baseline = model(x)
        with AttentionHookRecorder(model) as recorder:
            hooked = model(x)
    torch.testing.assert_close(baseline, hooked, atol=1e-7, rtol=1e-6)
    assert len(recorder.records) == 2
    assert recorder.records[0]["attention_probs"].shape == (2, 4, 21, 21)
