"""Explicit, hook-safe EEG Transformer and adapter construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util

import torch
from torch import nn


class ExplicitSelfAttention(nn.Module):
    """Non-fused MHA whose returned tensors are exactly those used by forward."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-2, -1)) * self.scale
        probs = logits.softmax(dim=-1)
        context = probs @ v
        output = self.proj(context.transpose(1, 2).reshape(batch, tokens, dim))
        return output, {"q": q, "k": k, "v": v, "attention_logits": logits, "attention_probs": probs}


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ExplicitSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward_with_internals(self, x):
        before = x
        attention_output, internals = self.attn(self.norm1(x))
        after = before + attention_output
        mlp_output = self.mlp(self.norm2(after))
        final = after + mlp_output
        return final, {
            **internals,
            "attention_output": attention_output,
            "residual_before_attention": before,
            "residual_after_attention": after,
            "mlp_output": mlp_output,
        }

    def forward(self, x):
        return self.forward_with_internals(x)[0]


class TinyEEGTransformer(nn.Module):
    """Architecture smoke-test model; never represented as pretrained."""

    def __init__(self, channels: int, n_times: int, patch_samples: int = 25, dim: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.channels, self.patch_samples = channels, patch_samples
        self.n_patches = n_times // patch_samples
        self.patch = nn.Linear(channels * patch_samples, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def embed(self, x):
        x = x[..., :self.n_patches * self.patch_samples]
        patches = x.unfold(-1, self.patch_samples, self.patch_samples)
        patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], self.n_patches, -1)
        tokens = self.patch(patches)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1) + self.pos

    def forward_with_internals(self, x):
        embedded = self.embed(x)
        current, layers = embedded, []
        for block in self.blocks:
            current, info = block.forward_with_internals(current)
            layers.append(info)
        final = self.norm(current)
        return {
            "input_embedding": embedded,
            "layers": layers,
            "final_tokens": final,
            "pooled_representation": final[:, 0],
        }

    def forward(self, x):
        return self.forward_with_internals(x)["pooled_representation"]


class AttentionHookRecorder:
    """Non-mutating forward hooks for auditing the actual attention calls."""

    def __init__(self, model: TinyEEGTransformer):
        self.model = model
        self.records: list[dict[str, torch.Tensor]] = []
        self.handles = []

    def __enter__(self):
        def capture(module, inputs, output):
            del module, inputs
            _, internals = output
            self.records.append(internals)

        self.handles = [block.attn.register_forward_hook(capture) for block in self.model.blocks]
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class EEGTransformerAdapter:
    def __init__(self, model: nn.Module, model_kind: str, pretrained: bool, provenance: str):
        self.model = model.cpu().eval()
        self.model_kind = model_kind
        self.pretrained = pretrained
        self.provenance = provenance

    def forward_with_internals(self, x, metadata=None) -> dict[str, Any]:
        del metadata
        with torch.inference_mode():
            if isinstance(self.model, TinyEEGTransformer):
                with AttentionHookRecorder(self.model) as recorder:
                    result = self.model.forward_with_internals(x.cpu())
                if len(recorder.records) != len(result["layers"]):
                    raise RuntimeError("Attention hook did not observe every Transformer layer")
                # Identity checks ensure the exported tensors came from the
                # exact attention calls observed by hooks, not recomputation.
                for hooked, exported in zip(recorder.records, result["layers"]):
                    if hooked["q"].data_ptr() != exported["q"].data_ptr():
                        raise RuntimeError("Exported Q/K/V do not match hooked forward tensors")
                return result
            return self.model.forward_with_internals(x.cpu())


def build_adapter(config: dict[str, Any], channels: int, n_times: int, seed: int) -> tuple[EEGTransformerAdapter, dict[str, Any]]:
    model_cfg = config["model"]
    checkpoint = str(model_cfg.get("checkpoint") or "").strip()
    requested = str(model_cfg.get("type", "auto")).lower()
    cbramod_available = importlib.util.find_spec("cbramod") is not None
    # Official CBraMod checkpoints require the official package architecture.
    # We never silently load an arbitrary state dict into a different network.
    fallback_reason = ""
    if requested in {"auto", "cbramod"} and checkpoint and cbramod_available:
        raise NotImplementedError(
            "The installed CBraMod package must expose an explicit adapter mapping; "
            "set model.type=tiny for architecture smoke testing."
        )
    if requested == "cbramod":
        fallback_reason = "official CBraMod package/checkpoint unavailable"
    elif requested == "auto":
        fallback_reason = "no local official CBraMod checkpoint configured"
    torch.manual_seed(seed)
    model = TinyEEGTransformer(
        channels, n_times, int(model_cfg["patch_samples"]), int(model_cfg["embedding_dim"]),
        int(model_cfg["heads"]), int(model_cfg["layers"]),
    )
    if checkpoint:
        state = torch.load(Path(checkpoint), map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=True)
        adapter = EEGTransformerAdapter(model, "tiny_eeg_transformer", True, f"local checkpoint: {checkpoint}")
    else:
        adapter = EEGTransformerAdapter(model, "tiny_eeg_transformer", False, "randomly initialized architecture fallback")
    return adapter, {
        "requested_model": requested,
        "cbramod_package_available": cbramod_available,
        "checkpoint_configured": bool(checkpoint),
        "fallback_reason": fallback_reason,
    }
