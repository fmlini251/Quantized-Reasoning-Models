"""Qwen2 with every linear layer routed through the Ozaki custom GEMM (linear-only),
while attention (QK^T / softmax / PV) stays on vLLM's native paged FlashAttention.

This is the vLLM equivalent of `inference_transformers.py --linear_only`: q/k/v/o_proj,
gate/up/down_proj are emulated with ozaki1/ozaki1_fp; the attention score matmuls are
NOT (they remain a fused paged kernel), so we keep continuous batching + paged KV cache.

Selection (no model files edited on disk) -- set the HF config architectures and an
`ozaki_config` block via vLLM hf_overrides, e.g.::

    hf_overrides={
        "architectures": ["Qwen2OzakiForCausalLM"],
        "ozaki_config": {"nmp": 6, "chunk_size": 32, "rslt_type": "ozaki1_fp"},
    }

`inference_vllm.py --ozaki_placement linear_only ...` (or `full` for the full variant) wires this
up automatically.

Known divergence from the transformers --linear_only path: `lm_head` (a vLLM
ParallelLMHead, not a LinearBase) stays native here; the transformers path also
Ozaki-fies lm_head. lm_head runs once per generated token (not per layer) so the impact
is small, but it must be accounted for when validating numerical equivalence.
"""
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

logger = init_logger(__name__)

# Worker propagation for the "full" (ozaki attention) variant: when OZAKI_FULL_ATTENTION=1,
# importing this model module (driver via registry, spawned workers via model-qualname
# import) installs the custom ozaki attention backend before any Attention layer is built.
import os as _os
if _os.environ.get("OZAKI_FULL_ATTENTION") == "1":
    try:
        from vllm_custom.model_executor.layers.ozaki_attention import install_ozaki_attention_backend
        install_ozaki_attention_backend()
    except Exception as _e:  # never break linear-only / bf16 imports
        logger.warning("[Ozaki] full-attention backend install skipped: %s", _e)


class Qwen2OzakiForCausalLM(Qwen2ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        cfg = vllm_config.model_config.hf_config
        oz_cfg = getattr(cfg, "ozaki_config", None) or {}
        nmp = int(oz_cfg.get("nmp", 6))
        chunk_size = int(oz_cfg.get("chunk_size", 32))
        rslt_type = oz_cfg.get("rslt_type", "ozaki1_fp")
        # weight_cache=False re-encodes the weight every forward instead of holding the ~3x
        # B_fold cache -> only 1x bf16 weight resident, so nmp=6 fits a single GPU.
        use_weight_cache = bool(oz_cfg.get("weight_cache", True))
        # Ozaki-2 (RNS / CRT) params -- only consumed for rslt_type ozaki2 / ozaki2_fp; the
        # ozaki1 path ignores them. `s` (RNS modulus count) is required for ozaki2.
        s = oz_cfg.get("s", None)
        s = int(s) if s is not None else None
        scale_method = oz_cfg.get("scale_method", "new_compressed")
        shift_bits = int(oz_cfg.get("shift_bits", 7))
        m_frac_bits = int(oz_cfg.get("M_frac_bits", 8))

        # Lazy import: pulls in the compiled `emulation` CUDA stack, which is only needed
        # when an Ozaki model is actually instantiated (not at registry-import time).
        from vllm_custom.model_executor.layers.ozaki_linear import (
            OzakiLinearMethod, build_ozaki_configs)

        gcfg, oz = build_ozaki_configs(nmp, rslt_type, chunk_size, weight_cache=use_weight_cache,
                                       s=s, scale_method=scale_method, shift_bits=shift_bits,
                                       M_frac_bits=m_frac_bits)
        method = OzakiLinearMethod(gcfg, oz, use_weight_cache=use_weight_cache)

        n_routed = 0
        for _name, module in self.named_modules():
            # Only nn.Linear-equivalents are LinearBase: qkv_proj, o_proj, gate_up_proj,
            # down_proj. The attention score op (self.attn) is NOT a LinearBase, so it is
            # left untouched -> native paged attention == linear-only.
            if isinstance(module, LinearBase):
                module.quant_method = method
                module._ozaki_wc = None
                n_routed += 1

        scheme_param = f"s={s}" if rslt_type in ("ozaki2", "ozaki", "ozaki2_fp") else f"nmp={nmp}"
        logger.info(
            "[Ozaki] linear-only: routed %d linear layers through ozaki "
            "(rslt_type=%s, %s, chunk_size=%d, weight_cache=%s); attention + lm_head stay native.",
            n_routed, rslt_type, scheme_param, chunk_size, use_weight_cache)
