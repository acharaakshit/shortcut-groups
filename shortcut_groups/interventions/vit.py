import importlib
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from lxt.efficient import monkey_patch, monkey_patch_zennit
from zennit.core import BasicHook

from .common import (
    region_map_to_grid,
    spatial_intervention_scale,
)


PATCH_GRID = 14

# Spatial intervention scales act on patch tokens and leave CLS unchanged.
def _with_cls_token(patch_scale):
    return torch.cat((patch_scale.new_ones(1), patch_scale))


@contextmanager
def attention_lrp_patched(model):
    module_name = model.__class__.__module__
    # patch the three types of layers first based on LXT
    patched_classes = (torch.nn.GELU, torch.nn.LayerNorm, torch.nn.MultiheadAttention)
    # store the original forward
    original_forwards = {cls: cls.forward for cls in patched_classes}
    original_hook_forward = BasicHook.forward
    original_hook_backward = BasicHook.backward
    # monkey patching is done here 
    try:
        monkey_patch(importlib.import_module(module_name), verbose=False)
        monkey_patch_zennit(verbose=False)
        # now pause and compute the attnlrp before the restoration of original forwards
        yield
    finally:
        # restore the original forwards now
        for cls, forward in original_forwards.items():
            cls.forward = forward
        BasicHook.forward = original_hook_forward
        BasicHook.backward = original_hook_backward

# suppression applied to the feature resolutions of different vit layers/blocks
def vit_intervention_token_scale(
    shortcut,
    task,
    intervention,
    seed=0,
):
    patch_shortcut = region_map_to_grid(shortcut, PATCH_GRID, PATCH_GRID).flatten()
    patch_task = region_map_to_grid(task, PATCH_GRID, PATCH_GRID).flatten()
    patch_scale = spatial_intervention_scale(
        patch_shortcut,
        patch_task,
        intervention,
        seed=seed,
    )
    return _with_cls_token(patch_scale)

# groups of layers selected for our target ablation analysis
def _selected_layers(model, layer_mode):
    layers = list(model.encoder.layers)
    layer_slice = {
        "all": slice(None),
        "last": slice(-1, None),
        "last3": slice(-3, None),
        "last6": slice(-6, None),
    }[layer_mode]
    return layers[layer_slice]

# obtain key, query and value vectors from the attention module of the specific layer provided to the function
def _attention_values(
    module,
    query,
    key,
    value,
):
    batch, tokens, embed_dim = query.shape
    q_weight, k_weight, v_weight = module.in_proj_weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = module.in_proj_bias.chunk(3, dim=0)
    head_dim = embed_dim // module.num_heads
    # Match OSCAR's CP-LRP rule: relevance flows through values, not Q/K.
    q = F.linear(query.detach(), q_weight, q_bias).view(
        batch, tokens, module.num_heads, head_dim
    ).transpose(1, 2)
    k = F.linear(key.detach(), k_weight, k_bias).view(
        batch, tokens, module.num_heads, head_dim
    ).transpose(1, 2)
    v = F.linear(value, v_weight, v_bias).view(
        batch, tokens, module.num_heads, head_dim
    ).transpose(1, 2)
    return q, k, v


def _scaled_attention_forward(module, token_scale, targets):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        
        batch, tokens, embed_dim = query.shape
        q, k, v = _attention_values(self, query, key, value)

        scale = _expand_token_scale(token_scale, batch, tokens).to(
            device=v.device,
            dtype=v.dtype,
        ).view(batch, 1, tokens, 1)

        # in target block analysis, we scale Q/K/V for ablations
        if "query" in targets:
            q = q * scale
        if "key" in targets:
            k = k * scale
        if "value" in targets:
            v = v * scale

        head_dim = embed_dim // self.num_heads
        attention = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        if attn_mask is not None:
            attention = attention + attn_mask
        if key_padding_mask is not None:
            attention = attention.masked_fill(
                key_padding_mask[:, None, None, :].bool(),
                float("-inf"),
            )
        attention = torch.softmax(attention, dim=-1)
        attention = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )

        out = attention @ v
        out = out.transpose(1, 2).reshape(batch, tokens, embed_dim)
        return self.out_proj(out), None

    return MethodType(forward, module)

# batched token scale which might be needed for shortcut and dataset level suppression
def _expand_token_scale(token_scale, batch, tokens):
    if token_scale.ndim == 1:
        return token_scale.view(1, tokens).expand(batch, tokens)
    return token_scale

# independent of Q,K,V scaling, here we have MLP level intervention
def _scaled_mlp_forward(module, token_scale):
    original_forward = module.forward

    def forward(self, inputs):
        output = original_forward(inputs)
        scale = _expand_token_scale(
            token_scale,
            output.shape[0],
            output.shape[1],
        ).to(device=output.device, dtype=output.dtype)
        return output * scale.unsqueeze(-1)

    return MethodType(forward, module)


@contextmanager
def patch_vit(model, token_scale, targets=("value",), layer_mode="all"):
    targets = set(targets)
    unknown = targets - {"query", "key", "value", "mlp"}
    if unknown:
        raise ValueError(f"Unknown ViT intervention targets: {sorted(unknown)}")

    missing = object()
    originals = []
    try:
        for layer in _selected_layers(model, layer_mode):
            if targets & {"query", "key", "value"}:
                attention = layer.self_attention
                originals.append((
                    attention,
                    attention.__dict__.get("forward", missing),
                ))
                attention.forward = _scaled_attention_forward(
                    attention,
                    token_scale,
                    targets,
                )
            if "mlp" in targets:
                mlp = layer.mlp
                originals.append((
                    mlp,
                    mlp.__dict__.get("forward", missing),
                ))
                mlp.forward = _scaled_mlp_forward(mlp, token_scale)
        # again, apply intervention, pause to compute the performance and restore original model
        yield
    finally:
        for module, original in reversed(originals):
            if original is missing:
                module.__dict__.pop("forward", None)
            else:
                module.forward = original
