"""PROTOTYPE: Ozaki custom-GEMM attention backend for vLLM (the "full" variant).

This makes the attention score matmuls (QK^T and attn.V) also go through the Ozaki int8
decomposition -- i.e. `inference_transformers.py` WITHOUT `--linear_only` -- inside vLLM. It
subclasses the XFormers backend (reusing its paged-KV cache layout, metadata, and
metadata-builder) and overrides ONLY `AttentionImpl.forward` to compute attention eagerly
with `batched_gemm` (mirrors emulation.llm.ozaki_qwen.custom_qwen2_eager_attention_forward).

Status: nmp=1 single-GPU PROTOTYPE for plumbing/numerical validation. Slow (eager, per-seq
decode loop, no flash/paging-kernel); correctness-first, not speed. Combine with
Qwen2OzakiForCausalLM (linear-only) to get the full all-ozaki model.

Three integration pieces:
  1. backend selection  -> install_ozaki_attention_backend() monkeypatches
     vllm.attention.selector.get_attn_backend to return OzakiAttentionBackend. Must run in
     EVERY process that builds attention layers (driver + spawned workers), so we also honor
     the env var OZAKI_FULL_ATTENTION at import time.
  2. forward override    -> OzakiAttentionImpl.forward (below)
  3. paged-KV gather (decode) -> _gather_kv_from_cache (inverts the PagedAttention layout)

Vendors nothing from emulation.llm.utils/ozaki_llama (transformers-version-skew safe); uses
only emulation.llm.ozaki_matmul.batched_gemm (lazy import).
"""
import os
from typing import Optional

import torch
import torch.nn as nn

# Set by install_ozaki_attention_backend(); read lazily when the first impl is built.
_OZAKI_ATTN_PARAMS = None  # dict(nmp=, chunk_size=, rslt_type=)

# Decode attention processes seqs in mini-batches so the pad-to-max KV gather + scores stay
# bounded regardless of context length: each mini-batch keeps (n_seqs * pad_len) <= this many
# tokens. At 256k budget and 32k context -> ~8 seqs/batch -> ~1GB transient (comfortably
# fits alongside the native KV pool even at gpu_memory_utilization 0.85). Override via env
# OZAKI_DECODE_ATTN_TOKEN_BUDGET.
_DECODE_ATTN_TOKEN_BUDGET = int(os.environ.get("OZAKI_DECODE_ATTN_TOKEN_BUDGET", 1 << 18))

# One-time-per-process flag so we log (at WARNING, which propagates from TP workers) that the
# ozaki attention impl is actually executing in this process — used to verify TP propagation.
_OZAKI_ATTN_ANNOUNCED = False


def set_ozaki_attention_params(nmp, chunk_size=32, rslt_type="ozaki1_fp", s=None,
                               scale_method="new_compressed", shift_bits=7, M_frac_bits=8,
                               gemm_bits=8, byte_split_style="all_signed_clamp_pos"):
    global _OZAKI_ATTN_PARAMS
    _OZAKI_ATTN_PARAMS = {
        "nmp": int(nmp), "chunk_size": int(chunk_size), "rslt_type": rslt_type,
        # Ozaki-2 (RNS) params; only consumed for rslt_type ozaki2 / ozaki2_fp.
        "s": (int(s) if s is not None else None), "scale_method": scale_method,
        "shift_bits": int(shift_bits), "M_frac_bits": int(M_frac_bits),
        # Ozaki-1 GEMM unit bit-width w (8 = int8 byte-split; 2/4 = w-bit emulation).
        "gemm_bits": int(gemm_bits),
        # Ozaki-1 integer digit-split (chunk) method.
        "byte_split_style": byte_split_style,
    }


def _resolve_params():
    if _OZAKI_ATTN_PARAMS is not None:
        return _OZAKI_ATTN_PARAMS
    # Fallback to env (so spawned workers that re-import this module still get the config).
    _s = os.environ.get("OZAKI_ATTN_S", "")
    return {
        "nmp": int(os.environ.get("OZAKI_ATTN_NMP", "6")),
        "chunk_size": int(os.environ.get("OZAKI_ATTN_CHUNK", "32")),
        "rslt_type": os.environ.get("OZAKI_ATTN_RSLT", "ozaki1_fp"),
        "s": (int(_s) if _s not in ("", "None") else None),
        "scale_method": os.environ.get("OZAKI_ATTN_SCALE_METHOD", "new_compressed"),
        "shift_bits": int(os.environ.get("OZAKI_ATTN_SHIFT_BITS", "7")),
        "M_frac_bits": int(os.environ.get("OZAKI_ATTN_M_FRAC_BITS", "8")),
        "gemm_bits": int(os.environ.get("OZAKI_ATTN_GEMM_BITS", "8")),
        "byte_split_style": os.environ.get("OZAKI_ATTN_BYTE_SPLIT_STYLE", "all_signed_clamp_pos"),
    }


def _pad_reduction_to_chunk(A, B, chunk_size):
    """Zero-pad the shared reduction dim (A's last dim, B's 2nd-last dim) up to ``chunk_size``
    when it is smaller, then return (A, B).

    The ozaki encode kernels use ``tl.arange(0, cs)`` with ``cs = min(reduction, chunk_size)``,
    which Triton requires to be a power of two. For linear layers the reduction dim is large so
    ``cs == chunk_size`` (a power of two), but the attention PV reduction is ``kv_len``, which can
    be a small non-power-of-two -- e.g. vLLM's short memory-profile sequences, or a prompt
    shorter than ``chunk_size`` -- giving ``cs == kv_len`` and crashing the kernel
    ("arange's range must be a power of 2"). Padding the reduction with zeros is numerically
    EXACT (the extra products are 0), and makes ``cs == chunk_size``. When reduction >=
    chunk_size, batched_gemm already pads it to a multiple of chunk_size internally, so we
    leave it untouched.
    """
    r = A.shape[-1]
    if r >= chunk_size:
        return A, B
    pad = chunk_size - r
    return (torch.nn.functional.pad(A, (0, pad)),
            torch.nn.functional.pad(B, (0, 0, 0, pad)))


def _ozaki_eager_attention(query, key, value, scaling, gcfg, oz, base_name):
    """query/key/value: [n, q_len, head_dim] / [n, kv_len, head_dim] (n = batch*heads,
    KV already repeated to n). Causal mask is applied by the callers via the q/kv length
    alignment (decode: q_len=1 attends to all kv; prefill: handled per-seq with a mask).
    Returns [n, q_len, head_dim]. Mirrors custom_qwen2_eager_attention_forward's two
    batched_gemm calls + fp32 softmax."""
    import copy
    from emulation.llm.ozaki_matmul import batched_gemm

    qk_cfg = copy.copy(gcfg); qk_cfg.name = base_name + ".attn_weights"
    # QK^T reduction = head_dim (normally a power of two >= chunk_size, so a no-op); padded
    # defensively for chunk_size > head_dim.
    A, B = _pad_reduction_to_chunk(query, key.transpose(1, 2), gcfg.chunk_size)
    attn = batched_gemm(A, B, custom_gemm_config=qk_cfg,
                        ozaki_config=oz, out_dtype=query.dtype).to(query.dtype) * scaling
    return attn  # caller adds mask + softmax, then calls _ozaki_pv


def _ozaki_pv(attn_weights, value, gcfg, oz, base_name):
    import copy
    from emulation.llm.ozaki_matmul import batched_gemm
    pv_cfg = copy.copy(gcfg); pv_cfg.name = base_name + ".attn_output"
    # PV reduction = kv_len, which can be a small non-power-of-two -> pad up to chunk_size.
    aw, v = _pad_reduction_to_chunk(attn_weights, value, gcfg.chunk_size)
    return batched_gemm(aw, v, custom_gemm_config=pv_cfg,
                        ozaki_config=oz, out_dtype=attn_weights.dtype).to(attn_weights.dtype)


def _gather_kv_from_cache(key_cache, value_cache, block_table, seq_len, num_kv_heads, head_size):
    """Invert the PagedAttention cache layout into contiguous [seq_len, num_kv_heads, head_size].
    key_cache:   [num_blocks, num_kv_heads, head_size//x, block_size, x]
    value_cache: [num_blocks, num_kv_heads, head_size, block_size]
    """
    block_size = value_cache.shape[-1]
    pos = torch.arange(seq_len, device=key_cache.device)
    blk = block_table[pos // block_size]
    off = pos % block_size
    k = key_cache[blk, :, :, off, :]            # [L, kv_heads, head//x, x]
    k = k.reshape(seq_len, num_kv_heads, head_size)
    v = value_cache[blk, :, :, off]             # [L, kv_heads, head]
    v = v.reshape(seq_len, num_kv_heads, head_size)
    return k, v


def install_ozaki_attention_backend(nmp=None, chunk_size=32, rslt_type="ozaki1_fp", s=None,
                                    scale_method="new_compressed", shift_bits=7, M_frac_bits=8,
                                    gemm_bits=8, byte_split_style="all_signed_clamp_pos"):
    """Monkeypatch vLLM's attention-backend selector to return OzakiAttentionBackend.
    Call BEFORE the LLM is built (attention layers resolve the backend at construction)."""
    if nmp is not None:
        set_ozaki_attention_params(nmp, chunk_size, rslt_type, s, scale_method, shift_bits,
                                   M_frac_bits, gemm_bits, byte_split_style)
        os.environ["OZAKI_ATTN_NMP"] = str(nmp)
        os.environ["OZAKI_ATTN_CHUNK"] = str(chunk_size)
        os.environ["OZAKI_ATTN_RSLT"] = rslt_type
        # Ozaki-2 (RNS) params propagated to spawned TP workers (consumed for ozaki2 / ozaki2_fp).
        os.environ["OZAKI_ATTN_S"] = ("" if s is None else str(s))
        os.environ["OZAKI_ATTN_SCALE_METHOD"] = scale_method
        os.environ["OZAKI_ATTN_SHIFT_BITS"] = str(shift_bits)
        os.environ["OZAKI_ATTN_M_FRAC_BITS"] = str(M_frac_bits)
        os.environ["OZAKI_ATTN_GEMM_BITS"] = str(gemm_bits)
        os.environ["OZAKI_ATTN_BYTE_SPLIT_STYLE"] = byte_split_style
    os.environ["OZAKI_FULL_ATTENTION"] = "1"

    def _patched(*args, **kwargs):
        return OzakiAttentionBackend

    # Patch both the selector module AND the names already imported-by-value into the
    # attention layer module (layer.py does `from ...selector import get_attn_backend`).
    import vllm.attention.selector as sel
    sel.get_attn_backend = _patched
    sel._cached_get_attn_backend = _patched
    import vllm.attention.layer as lyr
    lyr.get_attn_backend = _patched


# Apply at import in spawned workers (driver calls install_* explicitly).
if os.environ.get("OZAKI_FULL_ATTENTION") == "1":
    try:
        install_ozaki_attention_backend()
    except Exception:
        pass


from vllm.attention.backends.xformers import XFormersBackend, XFormersImpl
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.utils import get_num_prefill_decode_query_kv_tokens
from vllm.attention.ops.paged_attn import PagedAttention


class OzakiAttentionBackend(XFormersBackend):
    @staticmethod
    def get_name() -> str:
        # Report XFORMERS so vLLM's _Backend-enum mapping (backend_name_to_enum) and the
        # downstream `== _Backend.XFORMERS` checks treat us like the xformers backend we
        # subclass. The ozaki behavior comes from get_impl_cls() below.
        return "XFORMERS"

    @staticmethod
    def get_impl_cls():
        return OzakiAttentionImpl


class OzakiAttentionImpl(XFormersImpl):
    """XFormersImpl but QK^T / softmax / PV run through the Ozaki custom GEMM.
    Eager + per-seq (correctness-first prototype). Decoder self-attention only."""

    def _ensure_cfg(self):
        if getattr(self, "_oz_gcfg", None) is None:
            from vllm_custom.model_executor.layers.ozaki_linear import build_ozaki_configs
            p = _resolve_params()
            self._oz_gcfg, self._oz_oz = build_ozaki_configs(
                p["nmp"], p["rslt_type"], p["chunk_size"],
                s=p.get("s"), scale_method=p.get("scale_method", "new_compressed"),
                shift_bits=p.get("shift_bits", 7), M_frac_bits=p.get("M_frac_bits", 8),
                gemm_bits=p.get("gemm_bits", 8),
                byte_split_style=p.get("byte_split_style", "all_signed_clamp_pos"))
            global _OZAKI_ATTN_ANNOUNCED
            if not _OZAKI_ATTN_ANNOUNCED:
                _OZAKI_ATTN_ANNOUNCED = True
                # WARNING level so it propagates from spawned TP workers (per-process, once).
                import logging
                logging.getLogger("vllm").warning(
                    "[Ozaki] OzakiAttentionImpl ACTIVE in pid=%d (nmp=%s, rslt=%s) -- "
                    "attention QK^T/PV via ozaki", os.getpid(), p["nmp"], p["rslt_type"])

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        assert self.attn_type == AttentionType.DECODER, \
            "OzakiAttentionImpl supports decoder self-attention only"
        self._ensure_cfg()
        gcfg, oz = self._oz_gcfg, self._oz_oz
        nH, nKV, hd = self.num_heads, self.num_kv_heads, self.head_size
        groups = nH // nKV
        dev = query.device
        NEG = torch.finfo(torch.float32).min

        query = query.view(-1, nH, hd)
        key = key.view(-1, nKV, hd)
        value = value.view(-1, nKV, hd)

        if kv_cache.numel() > 0:
            key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, nKV, hd)
            PagedAttention.write_to_paged_cache(
                key, value, key_cache, value_cache, attn_metadata.slot_mapping,
                self.kv_cache_dtype, layer._k_scale, layer._v_scale)

        nq, nkv, nd = get_num_prefill_decode_query_kv_tokens(attn_metadata, self.attn_type)
        out = torch.empty_like(query)

        # ===== Prefill: vectorized over ALL prefill seqs (pad-to-max + masked), one
        # batched_gemm for QK^T and one for PV -- no per-seq Python loop. =====
        pm = attn_metadata.prefill_metadata
        if pm is not None and pm.num_prefills > 0:
            P = pm.num_prefills
            qsl = pm.query_start_loc[:P + 1].to(torch.long)
            lens = qsl[1:] - qsl[:P]                                       # [P]
            Lm = int(lens.max().item())
            seq_id = torch.repeat_interleave(torch.arange(P, device=dev), lens)        # [nq]
            posn = torch.arange(nq, device=dev) - qsl[:P].index_select(0, seq_id)      # [nq]
            qp = query.new_zeros(P, Lm, nH, hd);  qp[seq_id, posn] = query[:nq]
            kp = key.new_zeros(P, Lm, nKV, hd);   kp[seq_id, posn] = key[:nkv]
            vp = value.new_zeros(P, Lm, nKV, hd); vp[seq_id, posn] = value[:nkv]
            q = qp.permute(0, 2, 1, 3).reshape(P * nH, Lm, hd)
            k = kp.repeat_interleave(groups, 2).permute(0, 2, 1, 3).reshape(P * nH, Lm, hd)
            v = vp.repeat_interleave(groups, 2).permute(0, 2, 1, 3).reshape(P * nH, Lm, hd)
            aw = _ozaki_eager_attention(q, k, v, self.scale, gcfg, oz, layer.layer_name)
            aw = aw.view(P, nH, Lm, Lm).float()
            ar = torch.arange(Lm, device=dev)
            allow = (ar[None, :] <= ar[:, None])[None] & (ar[None, :] < lens[:, None])[:, None, :]  # [P,Lm,Lm]
            aw = aw.masked_fill(~allow[:, None], NEG)
            aw = nn.functional.softmax(aw, dim=-1).to(q.dtype).view(P * nH, Lm, Lm)
            o = _ozaki_pv(aw, v, gcfg, oz, layer.layer_name).view(P, nH, Lm, hd).permute(0, 2, 1, 3)
            out[:nq] = o[seq_id, posn]

        # ===== Decode: gather each seq's K/V from the paged cache + GQA-grouped batched_gemm,
        # processed in MINI-BATCHES of seqs so the pad-to-max gather/scores stay bounded at
        # long context (avoids the [D * pad_len] OOM that killed the 32k run). Within a
        # mini-batch it's still one batched_gemm per QK^T/PV (no per-seq Python loop). =====
        dm = attn_metadata.decode_metadata
        if dm is not None and nd > 0:
            D = nd
            dq_all = query[nq:].reshape(D, nH, hd)
            seq_lens_all = dm.seq_lens_tensor.to(torch.long)              # [D]
            block_tables_all = dm.block_tables                            # [D, max_blocks]
            block_size = value_cache.shape[-1]
            max_blk = block_tables_all.shape[1]
            Lm_all = int(seq_lens_all.max().item())
            # seqs per mini-batch so (chunk * Lm_all) <= token budget -> bounded transient.
            chunk = max(1, _DECODE_ATTN_TOKEN_BUDGET // max(Lm_all, 1))
            for s0 in range(0, D, chunk):
                e0 = min(s0 + chunk, D)
                d = e0 - s0
                dq = dq_all[s0:e0]
                seq_lens = seq_lens_all[s0:e0]
                block_tables = block_tables_all[s0:e0]
                Lm = int(seq_lens.max().item())                           # pad to THIS batch's max
                posn = torch.arange(Lm, device=dev)[None, :].expand(d, Lm)
                valid = posn < seq_lens[:, None]
                blk = torch.gather(block_tables, 1, (posn // block_size).clamp(max=max_blk - 1))
                off = posn % block_size
                bf, of = blk.reshape(-1), off.reshape(-1)
                K = key_cache[bf, :, :, of, :].reshape(d, Lm, nKV, hd)     # invert paged layout, batched
                V = value_cache[bf, :, :, of].reshape(d, Lm, nKV, hd)
                # GQA-grouped: keep K/V at nKV heads (do NOT repeat to nH); treat each KV
                # head's `groups` query heads as the GEMM row dim (identical, nH/nKV less mem).
                q = dq.reshape(d, nKV, groups, hd).reshape(d * nKV, groups, hd)
                k = K.permute(0, 2, 1, 3).reshape(d * nKV, Lm, hd)
                v = V.permute(0, 2, 1, 3).reshape(d * nKV, Lm, hd)
                aw = _ozaki_eager_attention(q, k, v, self.scale, gcfg, oz, layer.layer_name)
                aw = aw.view(d, nKV, groups, Lm).float().masked_fill(~valid[:, None, None, :], NEG)
                aw = nn.functional.softmax(aw, dim=-1).to(q.dtype).reshape(d * nKV, groups, Lm)
                o = _ozaki_pv(aw, v, gcfg, oz, layer.layer_name)
                out[nq + s0:nq + e0] = o.view(d, nKV, groups, hd).reshape(d, nH, hd)

        return out.view(-1, nH * hd)
