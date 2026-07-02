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
                               gemm_bits=8, byte_split_style="all_signed_clamp_pos",
                               nmp_overrides=None):
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
        # Per-op nmp overrides {regex-pattern: nmp}. Attention resolves them against the op names
        # "<layer>.attn_weights" (QK^T) and "<layer>.attn_output" (P@V) -> attn_score and attn_output
        # can use different nmp. None => base nmp for both.
        "nmp_overrides": (dict(nmp_overrides) if nmp_overrides else None),
    }


def _resolve_params():
    if _OZAKI_ATTN_PARAMS is not None:
        return _OZAKI_ATTN_PARAMS
    # Fallback to env (so spawned workers that re-import this module still get the config).
    import json
    _s = os.environ.get("OZAKI_ATTN_S", "")
    _ov = os.environ.get("OZAKI_ATTN_NMP_OVERRIDES", "")
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
        "nmp_overrides": (json.loads(_ov) if _ov not in ("", "None") else None),
    }


def _resolve_nmp(name, overrides, base):
    """Per-op nmp: first override whose regex matches ``name`` wins, else ``base`` (mirrors the
    emulation batched_gemm dispatcher used by the eager path)."""
    if overrides:
        import re
        for pat, val in overrides.items():
            if re.search(pat, name):
                return int(val)
    return int(base)


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


def _attn_product(A, B, cfg, oz, out_dtype):
    """The QK^T / PV batched product. When env OZAKI_ATTN_EXACT=1, compute it as an EXACT bf16
    torch.matmul (NO Ozaki quantization) -- this isolates the eager backend's implementation
    gap vs native FlashAttention (the EAGER-EXACT control). Otherwise the normal Ozaki block-FP
    batched_gemm. The exact branch's zero-padded reduction dim is numerically a no-op, so the
    eager path is otherwise byte-identical between the two modes."""
    if os.environ.get("OZAKI_ATTN_EXACT", "0") == "1":
        return torch.matmul(A, B).to(out_dtype)
    from emulation.llm.ozaki_matmul import batched_gemm
    return batched_gemm(A, B, custom_gemm_config=cfg, ozaki_config=oz, out_dtype=out_dtype)


def _ozaki_eager_attention(query, key, value, scaling, gcfg, oz, base_name):
    """query/key/value: [n, q_len, head_dim] / [n, kv_len, head_dim] (n = batch*heads,
    KV already repeated to n). Causal mask is applied by the callers via the q/kv length
    alignment (decode: q_len=1 attends to all kv; prefill: handled per-seq with a mask).
    Returns [n, q_len, head_dim]. Mirrors custom_qwen2_eager_attention_forward's two
    batched_gemm calls + fp32 softmax."""
    import copy

    qk_cfg = copy.copy(gcfg); qk_cfg.name = base_name + ".attn_weights"
    # QK^T reduction = head_dim (normally a power of two >= chunk_size, so a no-op); padded
    # defensively for chunk_size > head_dim.
    A, B = _pad_reduction_to_chunk(query, key.transpose(1, 2), gcfg.chunk_size)
    attn = _attn_product(A, B, qk_cfg, oz, query.dtype).to(query.dtype) * scaling
    return attn  # caller adds mask + softmax, then calls _ozaki_pv


def _ozaki_pv(attn_weights, value, gcfg, oz, base_name):
    import copy
    pv_cfg = copy.copy(gcfg); pv_cfg.name = base_name + ".attn_output"
    # PV reduction = kv_len, which can be a small non-power-of-two -> pad up to chunk_size.
    aw, v = _pad_reduction_to_chunk(attn_weights, value, gcfg.chunk_size)
    return _attn_product(aw, v, pv_cfg, oz, attn_weights.dtype).to(attn_weights.dtype)


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
                                    gemm_bits=8, byte_split_style="all_signed_clamp_pos", flash=None,
                                    nmp_overrides=None, kv_cache_prefill=False):
    """Monkeypatch vLLM's attention-backend selector to return OzakiAttentionBackend.
    Call BEFORE the LLM is built (attention layers resolve the backend at construction).

    flash: None (default) leaves OZAKI_ATTN_FLASH as-is; True/False sets it. When enabled (and
    rslt_type==ozaki1_fp) attention runs through the Triton flash_ozaki kernel (online-softmax,
    non-cached, GQA-fold) instead of the eager batched_gemm path -- same ozaki1_fp math, ~e-3 apart.
    nmp_overrides: {regex: nmp} applied to the attention ops too -- match "attn_weights" (QK^T =
    attn_score) and/or "attn_output" (P@V) to give the two attention GEMMs different nmp.
    kv_cache_prefill: NOT IMPLEMENTED -- caching the ozaki digit-planes of K/V would multiply the
    KV-cache footprint by nD (3-5x for w4 nmp9-16), prohibitive for vLLM's KV-capacity-bound
    throughput, so it is intentionally blocked (raises NotImplementedError). Use non-cached flash."""
    import json
    if kv_cache_prefill:
        raise NotImplementedError(
            "kv_cache_prefill (caching ozaki K/V digit-planes for reuse across decode steps) is not "
            "implemented: it would multiply the vLLM KV cache by nD (the number of digit planes -- "
            "e.g. 4x for w4 nmp10, 5x for w4 nmp15), and since vLLM throughput is bound by KV-cache "
            "capacity (concurrency x context) that nD x blowup is a net loss versus the ~1.3x per-op "
            "decode gain. Use the default non-cached flash path (--ozaki_flash) instead.")
    if flash is not None:
        os.environ["OZAKI_ATTN_FLASH"] = "1" if flash else "0"
    if nmp is not None:
        set_ozaki_attention_params(nmp, chunk_size, rslt_type, s, scale_method, shift_bits,
                                   M_frac_bits, gemm_bits, byte_split_style, nmp_overrides)
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
        # per-op nmp overrides -> JSON env for spawned TP workers.
        os.environ["OZAKI_ATTN_NMP_OVERRIDES"] = (json.dumps(nmp_overrides) if nmp_overrides else "")
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
                byte_split_style=p.get("byte_split_style", "all_signed_clamp_pos"),
                nmp_overrides=p.get("nmp_overrides"))   # eager: batched_gemm resolves per-op by name
            self._oz_p = p
            self._oz_ovr = p.get("nmp_overrides")       # flash: resolved per-op in _flash()
            # Flash path (Triton flash_ozaki kernel): opt-in via OZAKI_ATTN_FLASH=1. The codegen kernel
            # is bit-faithful to production only for ozaki1_fp + all_signed_no_clamp (verified across
            # w=2/4/8); its clamp path diverges (~0.26 relerr) and ozaki2_fp/RNS is unsupported. Anything
            # else falls back to the eager batched_gemm path (faithful to every style) with a warning.
            _flash_req = os.environ.get("OZAKI_ATTN_FLASH", "0") == "1"
            _style = p.get("byte_split_style", "all_signed_clamp_pos")
            self._oz_flash = (_flash_req and p["rslt_type"] == "ozaki1_fp"
                              and _style == "all_signed_no_clamp")
            if _flash_req and not self._oz_flash:
                import logging
                logging.getLogger("vllm").warning(
                    "[Ozaki] OZAKI_ATTN_FLASH=1 but incompatible (rslt=%s, byte_split_style=%s) -- flash "
                    "requires ozaki1_fp + all_signed_no_clamp; using EAGER.", p["rslt_type"], _style)
            global _OZAKI_ATTN_ANNOUNCED
            if not _OZAKI_ATTN_ANNOUNCED:
                _OZAKI_ATTN_ANNOUNCED = True
                # WARNING level so it propagates from spawned TP workers (per-process, once).
                import logging
                logging.getLogger("vllm").warning(
                    "[Ozaki] OzakiAttentionImpl ACTIVE in pid=%d (nmp=%s, rslt=%s, mode=%s) -- "
                    "attention QK^T/PV via ozaki", os.getpid(), p["nmp"], p["rslt_type"],
                    "flash" if self._oz_flash else "eager")

    def _flash(self, q4, k4, v4, name, kv_lens=None, split_kv=False):
        """Ozaki1_fp Triton flash kernel. q4:[B,Hq,T,D]; k4,v4:[B,Hkv,N,D] (GQA folded internally).
        Causal online-softmax; kv_lens[B] masks padded decode positions. Returns [B,Hq,T,D].
        Per-op nmp: QK^T (attn_score) resolves nmp_overrides against "<name>.attn_weights", P@V
        (attn_output) against "<name>.attn_output" -- matching the eager batched_gemm op names.

        split_kv (decode only): use flash-decoding (flash_oz1fp_cg_splitkv) -- the B*Hkv decode grid is
        too small to fill the SMs, so the ALU-bound ozaki emulation can't hide behind the KV-read memory
        (ncu: decode ALU ~44%, DRAM ~30% vs exact 96%). Splitting the kv loop into n_splits program-z
        slices + an LSE combine raises occupancy; n_splits is auto-sized to ~oversubscribe the SMs. Same
        numerics as the non-split path (n_splits=1 is bit-identical)."""
        from flash_ozaki.flash_oz1fp_codegen import flash_oz1fp_cg
        p = self._oz_p; w = p["gemm_bits"]; cs = p["chunk_size"]
        nmp_qk = _resolve_nmp(name + ".attn_weights", self._oz_ovr, p["nmp"])
        nmp_pv = _resolve_nmp(name + ".attn_output", self._oz_ovr, p["nmp"])
        D = q4.shape[-1]
        if split_kv and q4.shape[2] == 1 and cs is not None and cs < D:   # decode + chunked path only
            import triton
            from flash_ozaki.flash_oz1fp_codegen import flash_oz1fp_cg_splitkv
            B, Hkv, N = q4.shape[0], k4.shape[1], k4.shape[2]
            nsm = torch.cuda.get_device_properties(q4.device).multi_processor_count
            zc = max(1, B * Hkv)
            n_tiles = max(1, -(-N // triton.next_power_of_2(cs)))         # ceil(N / BLOCK_N)
            n_splits = max(1, min(-(-16 * nsm // zc), n_tiles, 32))       # ~oversubscribe SMs, capped
            if n_splits > 1:
                return flash_oz1fp_cg_splitkv(q4, k4, v4, nmp=nmp_qk, w=w, nmp_pv=nmp_pv, w_pv=w,
                                              causal=True, sm_scale=self.scale, chunk_size=cs,
                                              byte_split_style=p["byte_split_style"], kv_lens=kv_lens,
                                              n_splits=n_splits)
        return flash_oz1fp_cg(q4, k4, v4, nmp=nmp_qk, w=w, nmp_pv=nmp_pv, w_pv=w, causal=True,
                              sm_scale=self.scale, chunk_size=cs,
                              byte_split_style=p["byte_split_style"], kv_lens=kv_lens)

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
            if self._oz_flash:
                # Flash: causal alone is correct here -- a valid query at posn<lens attends only kv<=posn
                # (all valid); padded query rows (posn>=lens) are computed but discarded by the index below.
                o4 = self._flash(qp.permute(0, 2, 1, 3), kp.permute(0, 2, 1, 3),
                                 vp.permute(0, 2, 1, 3), layer.layer_name)  # [P,nH,Lm,hd]
                out[:nq] = o4.permute(0, 2, 1, 3)[seq_id, posn]
            else:
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
                if self._oz_flash:
                    # Flash handles GQA fold internally; kv_lens masks the padded tail per seq (K/V beyond
                    # seq_len were gathered from clamped/garbage slots but n_end+mask never touch them).
                    o4 = self._flash(dq[:, :, None, :],                        # [d,nH,1,hd]
                                     K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3),  # [d,nKV,Lm,hd]
                                     layer.layer_name, kv_lens=seq_lens,       # [d,nH,1,hd]
                                     split_kv=True)                            # flash-decoding (occupancy)
                    out[nq + s0:nq + e0] = o4.reshape(d, nH, hd)
                else:
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
