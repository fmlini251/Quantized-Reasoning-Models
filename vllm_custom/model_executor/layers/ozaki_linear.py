"""Ozaki custom-GEMM LinearMethod for vLLM (linear-only emulation).

Routes every vLLM linear layer (QKV/MergedColumn/Row/Column ParallelLinear) through the
Ozaki int8-decomposition GEMM, while the attention score matmuls (QK^T / softmax / PV)
stay on vLLM's native paged FlashAttention.

This mirrors `inference_transformers.py --linear_only` but *inside* vLLM, so continuous batching,
the paged KV cache, and native attention are all preserved -- only the linear GEMM is
emulated. Selected via `Qwen2OzakiForCausalLM` (see qwen2_ozaki.py).

Both Ozaki schemes are supported via ``rslt_type`` (see ``build_ozaki_configs``):
  * ``ozaki1`` / ``ozaki1_fp`` -- block-FP polynomial, parameterized by ``nmp``.
  * ``ozaki2`` / ``ozaki2_fp`` -- RNS / CRT, parameterized by ``s``.

Dependency note
---------------
We depend ONLY on ``emulation.llm.ozaki_matmul`` (the compiled CUDA-extension GEMM, imports
with no `transformers` dependency) and the pure-numpy table ``emulation.oz2.table``. We
deliberately do NOT import ``emulation.llm.utils`` or ``emulation.llm.ozaki_llama``: those
pull heavy, version-sensitive top-level imports (transformers.masking_utils, auto_gptq, ...)
that are absent / skewed in the vLLM env (transformers 4.47 vs the emulation package's 4.57).
The three things we need from them are tiny and pure, so they are vendored below:
  * ``_prepare_ozaki`` builds the RNS / CRT tables for the ozaki2 config (copied from
    ``emulation/llm/utils.py::prepare_ozaki``; depends only on the pure-numpy oz2 table).
  * ``custom_matmul`` is copied verbatim from ``emulation/llm/ozaki_llama.py`` (it only calls
    ``batched_gemm`` + torch/copy/math). Keep in sync if the upstream changes.

The ``emulation`` import is done lazily inside the methods so importing this module never
fails in an env that lacks the CUDA stack (e.g. registry.py imported for a plain bf16 run).
"""
import copy
import math
from typing import Optional

import torch

from vllm.model_executor.layers.linear import LinearMethodBase


def _prepare_ozaki(s_lst, chunk_size):
    """Vendored from ``emulation/llm/utils.py::prepare_ozaki`` (pure: oz2 table + torch/numpy).

    Builds the RNS / CRT tables that ``GlobalOzaki2Config`` needs -- but only so its
    ``__init__`` can precompute the per-s ``M_exp`` / ``M_frac`` scalars from ``M_tensor``.
    The ozaki2_fp GEMM itself (``ozaki2_batched_gemm_fp``) never touches these RNS tensors on
    the GPU, so the config can stay on CPU exactly like the ozaki1 path. Vendored (not
    imported) for the same reason as ``_custom_matmul``: ``utils.py`` pulls
    transformers.masking_utils / auto_gptq at import, which are version-skewed / absent in the
    vLLM env. Keep in sync with the upstream. Returns (log2M, moduli, invM, NMi, M).
    """
    import numpy as np
    from emulation.oz2.table import oz2_table

    if 1 in s_lst:
        assert len(s_lst) == 1, "s=1 must be used alone (block-FP path, no RNS tables)"
        return None, None, None, None, None
    if s_lst == [None] or s_lst == []:
        return None, None, None, None, None
    max_s = max(s_lst)
    log2M = torch.tensor([oz2_table["simple_log2M"][s] for s in s_lst]).to(dtype=torch.float32)
    moduli = torch.tensor([int(oz2_table["moduli"][i][0]) for i in range(len(oz2_table["moduli"]))],
                          dtype=torch.int16)
    invM = torch.tensor([oz2_table["invM"][s] for s in s_lst], dtype=torch.float64)
    NMi = torch.zeros((len(s_lst), max_s, 2), dtype=torch.uint64)
    for i, s in enumerate(s_lst):
        NMi[i, :s, :] = torch.tensor(np.array(oz2_table["NMi_int64"][s]), dtype=torch.uint64)
    M = torch.tensor(np.array([oz2_table["M_int64"][s] for s in s_lst]))
    return log2M, moduli, invM, NMi, M


def build_ozaki_configs(nmp, rslt_type="ozaki1_fp", chunk_size=32, weight_cache=True,
                        s=None, scale_method="new_compressed", shift_bits=7, M_frac_bits=8,
                        nmp_overrides=None, s_overrides=None, gemm_bits=8,
                        byte_split_style="all_signed_clamp_pos"):
    """CustomGemmConfig + Global{Ozaki1,Ozaki2}Config for a single Ozaki GEMM.

    Kept in sync with `inference_ozaki.build_ozaki_configs` / the emulation's `evaluate_ppl.py`
    so the vLLM path emulates exactly what the transformers baselines do. ``rslt_type`` selects
    the scheme:
      * ``ozaki1`` / ``ozaki1_fp`` -- block-FP polynomial; parameterized by ``nmp``
        (GlobalOzaki1Config). ``s`` / ``scale_method`` / ``shift_bits`` / ``M_frac_bits`` are
        ignored.
      * ``ozaki2`` / ``ozaki2_fp`` -- RNS / CRT; parameterized by ``s`` (GlobalOzaki2Config),
        which needs the RNS tables from ``_prepare_ozaki``. ``ozaki2_fp`` is the bf16 byte-plane
        emulation (the validated path here), run through its fused Triton fold.

    `weight_cache=True` builds the encoded weight once and reuses it across forwards (both
    schemes); `weight_cache=False` re-encodes the weight on every forward (keeps only the 1x
    bf16 weight resident -- trades recompute for memory).

    Per-operation overrides (``nmp_overrides`` / ``s_overrides``): a {regex-pattern: value} map
    (emulation feature, see emulation/llm/config.py). At GEMM dispatch ``batched_gemm`` resolves
    the per-op nmp/s by ``re.search``-matching the layer name (``custom_gemm_config.name``, set
    per-layer by OzakiLinearMethod.apply) against each pattern; the first match wins, else the
    default ``nmp`` / ``s``. NOTE: vLLM FUSES q/k/v into ``qkv_proj`` and gate/up into
    ``gate_up_proj``, so the matchable linear-layer names are ``qkv_proj`` / ``o_proj`` /
    ``gate_up_proj`` / ``down_proj`` (+ ``layers.<i>`` for a specific decoder layer) -- the
    transformers-side ``q_proj`` / ``k_proj`` patterns do NOT exist here.
    """
    # NOTE: the emulation package was refactored (~2026-06) — GlobalOzakiConfig split into
    # GlobalOzakiConfigBase / GlobalOzaki1Config / GlobalOzaki2Config, and the ozaki1 config
    # dropped the RNS / scale_method / shift_bits kwargs (ozaki1 never used them).
    from emulation.llm.ozaki_matmul import CustomGemmConfig

    gcfg = CustomGemmConfig(
        in_feature_ts=4096, out_feature_ts=4096, chunk_size=chunk_size, name="",
        track_mtx_acc=False, track_model_acc=False, get_statistics=False, rslt_type=rslt_type,
    )
    if rslt_type in ("ozaki1", "ozaki1_fp"):
        # block-FP polynomial path: (nmp_lst, rounding, weight_cache).
        # nmp_lst stays a single default (the dispatcher asserts len==1 and resolves the per-op
        # nmp directly, so override values need not be pre-registered).
        # gemm_bits (=w): 8 = int8 byte-split / native fold; 2/4 = w-bit paths. For rslt_type
        # 'ozaki1' (int8 reference) w!=8 has no weight_cache, so re-encode every forward.
        if gemm_bits != 8 and rslt_type == "ozaki1" and weight_cache:
            raise ValueError("ozaki1 (int8) with gemm_bits!=8 has no weight_cache; pass "
                             "weight_cache=False, or use rslt_type='ozaki1_fp' for cached w!=8.")
        # byte_split_style: integer digit-split (chunk) method. 'all_signed_no_clamp' lets the
        # MSB digit reach +-2^(w-1), which only the bf16 emulation (ozaki1_fp) can represent.
        if byte_split_style == "all_signed_no_clamp" and rslt_type != "ozaki1_fp":
            raise ValueError("byte_split_style='all_signed_no_clamp' is ozaki1_fp only (the int8 "
                             "path cannot represent the +-2^(w-1) MSB digit).")
        from emulation.llm.ozaki_matmul import GlobalOzaki1Config
        oz = GlobalOzaki1Config(
            nmp_lst=[nmp], rounding="round_half_away_from_0",
            weight_cache=weight_cache,
            nmp_overrides=nmp_overrides,
            gemm_bits=gemm_bits,
            byte_split_style=byte_split_style,
        )
    elif rslt_type in ("ozaki2", "ozaki", "ozaki2_fp"):
        if gemm_bits != 8:
            raise ValueError(f"gemm_bits={gemm_bits} (w!=8) is an ozaki1 concept; rslt_type="
                             f"{rslt_type!r} (RNS / CRT) only supports w=8.")
        # RNS / CRT path. Mirrors evaluate_ppl.py's ozaki2 branch.
        if s is None:
            raise ValueError(f"rslt_type={rslt_type!r} (Ozaki-2 / RNS) requires an s value; got "
                             "s=None. Pass --ozaki_s (valid range 2..20).")
        from emulation.llm.ozaki_matmul import GlobalOzaki2Config
        # s_lst must hold the default s + every per-op override s (default first) so the RNS
        # tables for all of them exist and get_for_s can fetch each at dispatch -- mirrors
        # emulation/llm/config.py::build_global_config. Unlike ozaki1's nmp, ozaki2 looks the
        # resolved s up in s_lst, so override values MUST be registered here.
        s_ov = s_overrides or {}
        s_lst = list(dict.fromkeys([s] + [int(v) for v in s_ov.values()]))
        log2M_tensor, moduli_tensor, invM_tensor, NMi_tensor, M_tensor = _prepare_ozaki(s_lst, chunk_size)
        oz = GlobalOzaki2Config(
            s_lst=s_lst,
            log2M_tensor=log2M_tensor, moduli_tensor=moduli_tensor, NMi_tensor=NMi_tensor,
            M_tensor=M_tensor, invM_tensor=invM_tensor,
            rounding="round_half_away_from_0",
            shift_bits=shift_bits, M_frac_bits=M_frac_bits, weight_cache=weight_cache,
            s_overrides=s_ov,
        )
    else:
        raise ValueError(f"Unsupported ozaki rslt_type {rslt_type!r}; expected one of "
                         "ozaki1, ozaki1_fp, ozaki2, ozaki2_fp.")
    return gcfg, oz


def _custom_matmul(x, weight, custom_gemm_config, ozaki_config,
                   weight_cache=None, return_weight_cache=False, out_dtype=None):
    """Vendored verbatim from emulation/llm/ozaki_llama.py::custom_matmul (pure: only
    batched_gemm + torch/copy/math). See dependency note at top of file."""
    from emulation.llm.ozaki_matmul import batched_gemm

    base_name = custom_gemm_config.name
    _building_cache = return_weight_cache and weight_cache is None
    if _building_cache:
        built_cache = {}

    if weight is not None:
        out_features = weight.shape[2]
        in_features = weight.shape[1]
    else:
        assert weight_cache is not None, "weight=None requires weight_cache"
        meta = weight_cache['_meta']
        out_features = meta['out_features']
        in_features = meta['in_features']

    out_ts = custom_gemm_config.out_feature_ts if custom_gemm_config.out_feature_ts is not None else out_features
    in_ts = custom_gemm_config.in_feature_ts if custom_gemm_config.in_feature_ts is not None else in_features

    if weight is not None:
        w_out_chunks = weight.split(out_ts, dim=2)
        num_out_chunks = len(w_out_chunks)
    else:
        num_out_chunks = math.ceil(out_features / out_ts)

    y_out_chunks = []
    for out_idx in range(num_out_chunks):
        if weight is not None:
            w_out_chunk = w_out_chunks[out_idx]
            out_chunk_size = w_out_chunk.shape[2]
        else:
            out_start = out_idx * out_ts
            out_chunk_size = min(out_ts, out_features - out_start)

        out_name = base_name if num_out_chunks == 1 else f"{base_name}.out_chunk_{out_idx}"

        if weight is not None:
            w_in_chunks = w_out_chunk.split(in_ts, dim=1)
            num_in_chunks = len(w_in_chunks)
        else:
            num_in_chunks = math.ceil(in_features / in_ts)

        y_out_chunk = None if num_in_chunks == 1 else torch.zeros(
            *x.shape[:-1], out_chunk_size, dtype=torch.float32, device=x.device)

        x_in_chunks = x.split(in_ts, dim=-1)
        for in_idx in range(num_in_chunks):
            x_chunk = x_in_chunks[in_idx]
            w_chunk = w_in_chunks[in_idx] if weight is not None else None
            chunk_name = out_name if num_in_chunks == 1 else f"{out_name}.in_chunk_{in_idx}"
            if chunk_name == custom_gemm_config.name:
                cfg = custom_gemm_config
            else:
                cfg = copy.copy(custom_gemm_config)
                cfg.name = chunk_name
            chunk_key = (out_idx, in_idx)
            chunk_cache = weight_cache.get(chunk_key) if weight_cache is not None else None
            y_result = batched_gemm(x_chunk, w_chunk, cfg, ozaki_config,
                                    weight_cache=chunk_cache,
                                    return_weight_cache=return_weight_cache,
                                    out_dtype=out_dtype)
            if _building_cache and isinstance(y_result, tuple):
                y_chunk, chunk_built = y_result
                built_cache[chunk_key] = chunk_built
            else:
                y_chunk = y_result
            if y_out_chunk is None:
                y_out_chunk = y_chunk
            else:
                y_out_chunk += y_chunk
        y_out_chunks.append(y_out_chunk)
    out = y_out_chunks[0] if num_out_chunks == 1 else torch.cat(y_out_chunks, dim=2)
    if _building_cache:
        built_cache['_meta'] = {'out_features': out_features, 'in_features': in_features}
        return out, built_cache
    return out


class OzakiLinearMethod(LinearMethodBase):
    """Drop-in replacement for ``UnquantizedLinearMethod`` whose ``apply`` runs the GEMM
    through the Ozaki custom matmul.

    A single instance is shared across all linear layers (the configs are read-only); the
    per-layer encoded-weight cache lives on each layer as ``layer._ozaki_wc`` so QKV/MLP
    weights are encoded once and reused. The first ``apply`` for a layer builds the cache
    (it needs a live activation to run the encode+GEMM); subsequent calls hit the cache.
    """

    def __init__(self, gcfg, oz, use_weight_cache=True):
        self.gcfg = gcfg
        self.oz = oz
        # When False, re-encode the weight every forward instead of caching the ~3x B_fold.
        # Keeps only the 1x bf16 weight resident (does NOT free it after first apply).
        self.use_weight_cache = use_weight_cache
        # Per-op overrides resolve the nmp/s by re.search-matching the layer name against the
        # override patterns (in batched_gemm). That match key is custom_gemm_config.name, which
        # the shared gcfg leaves "" -- so we set it per layer in apply(), but ONLY when overrides
        # are configured, keeping the validated uniform path byte-identical (and avoiding a
        # per-call config copy) otherwise.
        self._has_overrides = bool(getattr(oz, "nmp_overrides", None) or
                                   getattr(oz, "s_overrides", None))

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        # We swap quant_method in *after* the layer is constructed, so the weight Parameter
        # already exists (created by UnquantizedLinearMethod). create_weights is therefore
        # never called on us in practice; delegate for completeness/safety.
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        UnquantizedLinearMethod().create_weights(
            layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, **extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        # The encoded-weight cache is built lazily on the first apply() (which has the
        # activation needed to run the encode+GEMM). Nothing to do at load time.
        return

    def apply(self, layer, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze = x.dim() == 2
        x3 = (x.unsqueeze(0) if squeeze else x).contiguous()  # (1, num_tokens, K)
        out_dtype = x.dtype
        # vLLM stores weight as (out_features, in_features) = (N, K); _custom_matmul wants (1, K, N).

        # With per-op overrides, hand _custom_matmul a gcfg whose .name is this layer's qualified
        # name (e.g. model.layers.0.self_attn.qkv_proj) so batched_gemm can resolve the per-op
        # nmp/s by pattern-matching it. Without overrides keep the shared gcfg (name="").
        gcfg = self.gcfg
        if self._has_overrides:
            name = getattr(layer, "_ozaki_name", None)
            if name:
                gcfg = copy.copy(self.gcfg)
                gcfg.name = name

        if not self.use_weight_cache:
            # No-cache mode: re-encode the weight every forward. The ~3x B_fold is built
            # transiently inside _custom_matmul and freed on return, so only the 1x bf16
            # weight stays resident -> nmp=6 fits a single GPU. Costs a per-forward fold
            # (O(N*K), cheap vs the O(M*N*K) GEMM; amortized over the token batch).
            w = layer.weight.to(x.dtype).t().unsqueeze(0).contiguous()
            y = _custom_matmul(x3, w, gcfg, self.oz,
                               return_weight_cache=False, out_dtype=out_dtype)
        else:
            wc = getattr(layer, "_ozaki_wc", None)
            if wc is not None:
                # Cache hit: weight side already encoded; only the activation is encoded here.
                y = _custom_matmul(x3, None, gcfg, self.oz,
                                   weight_cache=wc, out_dtype=out_dtype)
            else:
                # First call: encode the weight and stash the cache on the layer.
                w = layer.weight.to(x.dtype).t().unsqueeze(0).contiguous()
                result = _custom_matmul(x3, w, gcfg, self.oz,
                                        return_weight_cache=True, out_dtype=out_dtype)
                if isinstance(result, tuple):
                    y, layer._ozaki_wc = result
                    # Free the bf16 weight from VRAM: the cache-only path (weight=None) never
                    # touches it again. The encoded cache is ~3x the bf16 weight for ozaki1_fp,
                    # so keeping both would nearly double weight memory. Mirrors the
                    # transformers CustomCompressedLinear path ("weight may be on CPU").
                    layer.weight.data = layer.weight.data.to("cpu")
                else:
                    y = result

        y = y.to(out_dtype)
        if squeeze:
            y = y.squeeze(0)
        if bias is not None:
            y = y + bias
        return y
