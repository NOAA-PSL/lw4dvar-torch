"""Exact sliding-window attention without flash-attn, for AIFS processors.

AIFS checkpoints are trained with `attention_implementation: flash_attention`
and a sliding window (`window_size`, e.g. 512): query i attends to keys j with
|i - j| <= window_size along the hidden-mesh node order. flash-attn has no
working wheel for every platform (e.g. ROCm/MI300X here). anemoi's own
`scaled_dot_product_attention` fallback is exact too, but it forces the MATH
kernel on the full (seq x seq) score matrix -- ~4 GB per layer for the
1-degree checkpoint's 10944-node mesh with 16 heads, which does not fit a
multi-step 4D-Var rollout's backward pass.

`BlockSlidingWindowAttention` computes the same function block-locally: the
sequence is cut into blocks of `window_size` queries, and each block attends
to the keys from `window_size` before its start to `window_size` after its
end, masked to |i - j| <= window_size. Scores are then
(n_blocks x W x 3W) instead of (seq x seq), ~7x smaller at 10944 nodes.

With `fold_blocks=True` (default) the blocks are folded into the batch
dimension, so SDPA sees 4-D (batch, heads, seq, head_dim) inputs and can pick
a fused (memory-efficient) kernel; `fold_blocks=False` keeps the original
5-D layout, which leaves SDPA only its unfused MATH kernel.

`FlexSlidingWindowAttention` computes the same function with
`torch.nn.attention.flex_attention` (compiled Triton kernels, forward and
backward), which skips key blocks outside the window and never stores the
score matrix. Needs torch >= 2.5 and a working Triton (CUDA or ROCm).

`use_sliding_window_attention(model, impl)` swaps one of them ('block' or
'flex') into every attention layer of a loaded anemoi model. `install_flash_attn_stub()` lets a
checkpoint that pickled a `FlashAttentionWrapper` be unpickled where
flash-attn is not installed; the stub raises if it is ever called, so the
swap must follow the load.
"""

import math
import sys
import types

import torch
import torch.nn.functional as F


class BlockSlidingWindowAttention(torch.nn.Module):
    """Drop-in for anemoi's FlashAttentionWrapper / SDPAAttentionWrapper.

    Inputs and output are (batch, heads, seq, head_dim), as anemoi's
    `MultiHeadSelfAttention.attention_computation` passes them.
    """

    def __init__(self, fold_blocks=True):
        super().__init__()
        self.fold_blocks = fold_blocks
        self._mask_key = None
        self._mask = None

    def _block_mask(self, seq_len, window, device):
        key = (seq_len, window, str(device))
        if self._mask_key != key:
            n_blocks = math.ceil(seq_len / window)
            q_idx = torch.arange(n_blocks * window, device=device).view(n_blocks, window, 1)
            starts = torch.arange(n_blocks, device=device).view(n_blocks, 1, 1) * window
            k_idx = starts - window + torch.arange(3 * window, device=device).view(1, 1, 3 * window)
            self._mask = ((q_idx - k_idx).abs() <= window) & (k_idx >= 0) & (k_idx < seq_len)
            self._mask_key = key
        return self._mask

    def forward(self, query, key, value, batch_size, causal=False, window_size=None,
                dropout_p=0.0, softcap=None, alibi_slopes=None):
        if causal or dropout_p or softcap or alibi_slopes is not None:
            raise NotImplementedError("BlockSlidingWindowAttention supports non-causal, dropout-free, "
                                      "softcap-free, alibi-free attention only")
        if window_size is None or window_size < 0:
            return F.scaled_dot_product_attention(query, key, value)
        b, h, s, d = query.shape
        w = int(window_size)
        n_blocks = math.ceil(s / w)
        pad_q = n_blocks * w - s
        q = F.pad(query, (0, 0, 0, pad_q)).view(b, h, n_blocks, w, d)
        # keys/values padded by w on the left and (pad_q + w) on the right, so
        # block i's window [i*w - w, i*w + 2w) is a fixed-size slice
        k = F.pad(key, (0, 0, w, pad_q + w)).unfold(2, 3 * w, w).transpose(-1, -2)
        v = F.pad(value, (0, 0, w, pad_q + w)).unfold(2, 3 * w, w).transpose(-1, -2)
        mask = self._block_mask(s, w, query.device)
        if not self.fold_blocks:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.reshape(b, h, n_blocks * w, d)[:, :, :s]
        # (b, h, n, ., d) -> (b*n, h, ., d); the mask (n, w, 3w) broadcasts over heads
        fold = lambda x: x.transpose(1, 2).reshape(b * n_blocks, h, x.shape[-2], d)
        mask = mask.unsqueeze(1).repeat(b, 1, 1, 1)
        out = F.scaled_dot_product_attention(fold(q), fold(k), fold(v), attn_mask=mask)
        out = out.view(b, n_blocks, h, w, d).transpose(1, 2)
        return out.reshape(b, h, n_blocks * w, d)[:, :, :s]


_compiled_flex_attention = None


class FlexSlidingWindowAttention(torch.nn.Module):
    """Same interface and function as BlockSlidingWindowAttention, computed
    with compiled flex_attention and a block-sparse sliding-window mask."""

    def __init__(self):
        super().__init__()
        self._mask_key = None
        self._mask = None

    def _block_mask(self, seq_len, window, device):
        from torch.nn.attention.flex_attention import create_block_mask
        key = (seq_len, window, str(device))
        if self._mask_key != key:
            def sliding_window(b, h, q_idx, kv_idx):
                return (q_idx - kv_idx).abs() <= window
            self._mask = create_block_mask(sliding_window, None, None, seq_len, seq_len, device=device)
            self._mask_key = key
        return self._mask

    def forward(self, query, key, value, batch_size, causal=False, window_size=None,
                dropout_p=0.0, softcap=None, alibi_slopes=None):
        global _compiled_flex_attention
        if causal or dropout_p or softcap or alibi_slopes is not None:
            raise NotImplementedError("FlexSlidingWindowAttention supports non-causal, dropout-free, "
                                      "softcap-free, alibi-free attention only")
        if window_size is None or window_size < 0:
            return F.scaled_dot_product_attention(query, key, value)
        if _compiled_flex_attention is None:
            from torch.nn.attention.flex_attention import flex_attention
            _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
        mask = self._block_mask(query.shape[-2], int(window_size), query.device)
        return _compiled_flex_attention(query, key, value, block_mask=mask)


SLIDING_WINDOW_IMPLS = {"block": BlockSlidingWindowAttention, "flex": FlexSlidingWindowAttention}


def use_sliding_window_attention(model, impl="block"):
    """Replace the attention function of every anemoi attention layer in
    `model` with `SLIDING_WINDOW_IMPLS[impl]`; returns how many were
    replaced."""
    cls = SLIDING_WINDOW_IMPLS[impl]
    n = 0
    for module in model.modules():
        if hasattr(module, "attention_implementation") and hasattr(module, "attention"):
            module.attention = cls()
            module.attention_implementation = f"{impl}_sliding_window"
            n += 1
    return n


def install_flash_attn_stub():
    """Make `import flash_attn` succeed when flash-attn is absent, so a
    checkpoint whose pickled model references `flash_attn.flash_attn_func`
    can be loaded. Returns True if the stub was installed."""
    try:
        import flash_attn  # noqa: F401
        return False
    except ImportError:
        pass

    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash-attn is not installed; call use_sliding_window_attention() after loading")

    root = types.ModuleType("flash_attn")
    root.__version__ = "0.0.0+stub"
    root.flash_attn_func = _unavailable
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_func = _unavailable
    layers = types.ModuleType("flash_attn.layers")
    rotary = types.ModuleType("flash_attn.layers.rotary")
    rotary.RotaryEmbedding = type("RotaryEmbedding", (torch.nn.Module,), {"forward": _unavailable})
    root.flash_attn_interface, root.layers, layers.rotary = interface, layers, rotary
    sys.modules.update({"flash_attn": root, "flash_attn.flash_attn_interface": interface,
                        "flash_attn.layers": layers, "flash_attn.layers.rotary": rotary})
    return True
