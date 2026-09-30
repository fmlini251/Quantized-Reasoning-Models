import torch

from . import fmt_bridge
from .fmt_bridge import fmt_lib


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


# --------------------------------------------------------------------------------------
# E1-0 calibration format-selection freeze (4bit_mixed_precision_experiment.md SS3.4)
# --------------------------------------------------------------------------------------
# Shared observe/select machinery mixed into both quantizers.  During an observe pass the
# quantizer is a pass-through that accumulates, per candidate format in its pool, the total
# quantization SSE over the calibration tensors it sees (unified MSE-both scale rule).
# finalize_observe() then picks the argmin format ONCE and freezes it as self.fmt_cfg, so the
# per-(layer,op,role) dtype is fixed BEFORE transform training / GPTQ (the "map is input, not
# output" rule) and never drifts per-step afterwards.

def _observe_start(q, pool, role, unify_scale="mse", site_key=""):
    q._observe = True
    q._fmt_pool = list(pool)
    q._fmt_role = role
    q._fmt_unify = unify_scale
    q._fmt_sse = {}
    q._site_key = site_key


def _observe_accumulate(q, x):
    if x.numel() == 0:
        return
    sse = fmt_lib.format_selection_sse(
        x.detach().reshape(-1, x.shape[-1]), q._fmt_pool,
        role=q._fmt_role, unify_scale=q._fmt_unify)
    for k, v in sse.items():
        q._fmt_sse[k] = q._fmt_sse.get(k, 0.0) + v


def _observe_finalize(q):
    """Pick argmin-SSE format, freeze it as q.fmt_cfg, return the selection record."""
    pool, sse = q._fmt_pool, q._fmt_sse
    best = min(pool, key=lambda k: sse.get(k, float("inf")))
    q.fmt_cfg = fmt_lib.QuantConfig(fmt_id=best, role=q._fmt_role)
    if hasattr(q, "_fmt_plan"):
        q._fmt_plan = None            # force the frozen-fmt weight path to re-plan on next call
    energy = max(sse.get("_energy", 0.0), 1e-30)
    rec = {"site_key": q._site_key, "role": q._fmt_role, "fmt_id": best,
           "relerr": {k: (sse.get(k, float("inf")) / energy) ** 0.5 for k in pool}}
    q._observe = False
    q._fmt_pool = q._fmt_sse = None
    return rec


def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max -1
    else:
        q_max, q_min = torch.tensor(2 ** bits - 1), 0
    return q_max, q_min


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(round_ste(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(round_ste(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


class ActivationQuantizer(torch.nn.Module):
    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''
    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, num_groups=1,
                 fmt_cfg=None):
        super(ActivationQuantizer, self).__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.groupsize = groupsize
        self.num_groups = num_groups
        self.lac = lac
        self._clip_ratio = clip_ratio
        self.fmt_cfg = fmt_cfg  # W14: lossless_444 fmt_lib grid; None -> original uniform path
        if self.lac:
            init_value = 4.
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor_a_max = torch.nn.Parameter(torch.ones((num_groups, ))*init_value, requires_grad=True)
            self.clip_factor_a_min = torch.nn.Parameter(torch.ones((num_groups, ))*init_value, requires_grad=True)

        self.enable = True
        self._observe = False          # E1-0 format-selection freeze (see _observe_* helpers)

    def forward(self, x):
        if self.bits == 16 or (not self.enable):
            return x
        if self._observe:              # observe pass: accumulate per-format SSE, pass through
            _observe_accumulate(self, x)
            return x
        init_shape = x.shape
        x = x.reshape(-1, self.num_groups, init_shape[-1] if self.groupsize == -1 else self.groupsize)
        fq_x = self.fake_quant(x)
        return fq_x.reshape(*init_shape)

    def fake_quant(self, x):
        x_dtype = x.dtype
        if self.fmt_cfg is not None:
            return self._fmt_fake_quant(x).to(x_dtype)
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, self.q_max.to(x)).to(x_dtype)  # TODO

    def _fmt_fake_quant(self, x):
        """fmt_lib grid path. Alpha double-clipping rule (4bit_mixed_precision_experiment.md
        SS3.2): lac/clip_ratio clipping is applied to the TENSOR here (differentiable clamp, so
        clip factors keep their gradient), then fmt_lib owns grid + scale on the clipped input."""
        if self.lac or self._clip_ratio is not None:
            xmax = x.amax(-1, keepdim=True).clamp(min=0)
            xmin = x.amin(-1, keepdim=True).clamp(max=0)
            if self.lac:
                xmax = xmax * self.sigmoid(self.clip_factor_a_max).reshape(1, -1, 1)
                xmin = xmin * self.sigmoid(self.clip_factor_a_min).reshape(1, -1, 1)
            else:
                xmax = xmax * self._clip_ratio
                xmin = xmin * self._clip_ratio
            x = torch.minimum(torch.maximum(x, xmin), xmax)
        return fmt_lib.fake_quant_ste(x, self.fmt_cfg)

    def get_scale_zero(self, x):
        q_max = self.q_max.to(x)
        init_shape = x.shape
        xmax, xmin = x.amax(-1, keepdim=True), x.amin(-1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
        # # if self.groupsize > 0:
        # #     assert x.shape[-1] % self.groupsize == 0
        # #     x = x.reshape((-1, self.groupsize))
        # #     # TODO: add padding
        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor_a_max).reshape(1, -1, 1)
            xmin = xmin * self.sigmoid(self.clip_factor_a_min).reshape(1, -1, 1)
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / q_max)
            scale[tmp] = 1
            scale = scale.repeat(1, 1, x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)

            scale = scale.repeat(1, 1, x.shape[-1]).reshape(init_shape)
            zero = zero.repeat(1, 1, x.shape[-1]).reshape(init_shape)

        return scale, zero


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

        self.enable = True
        self._observe = False          # E1-0 format-selection freeze (see _observe_* helpers)
        self.fmt_cfg = None

    def configure(
        self,
        bits, groupsize=-1, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        fmt_cfg=None
    ):
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.fmt_cfg = fmt_cfg  # W14: fmt_lib grid; None -> original uniform path
        self._fmt_plan = None
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16 or (not self.enable):
            return
        if getattr(self, "_observe", False):   # observe pass: accumulate per-format SSE on the
            _observe_accumulate(self, x)        # (transformed+clipped) weight, leave scale unset
            return                              # so quantize() stays a pass-through (not ready())
        if getattr(self, "fmt_cfg", None) is not None:
            # fmt path: freeze a ScalePlan (scales + mixfp4 choice map) from this tensor.
            # GPTQ calls this per group slice, then quantize() per column against the plan;
            # RTN/train call it on the full (transformed) weight each step. Note lwc clipping
            # is applied to the tensor in FlatQuantizedLinear.apply_wclip BEFORE this point
            # (alpha rule), and mse comes from the fmt config, not --gptq_mse.
            xf = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            self._fmt_plan = fmt_lib.compute_scales(xf, self.fmt_cfg)
            self.scale = self._fmt_plan.scales  # keeps ready() truthful
            self.zero = torch.zeros(1, device=x.device)
            return
        if self.groupsize != -1:
            x = x.reshape(-1, self.groupsize)

        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    def quantize(self, x):
        x_dtype = x.dtype
        if self.enable and self.ready() and self.bits < 16:
            init_shape = x.shape
            if getattr(self, "fmt_cfg", None) is not None:
                # fmt_lib handles blocking internally (groupsize reshape unnecessary: blocks
                # run along the last dim and never cross tp-shard boundaries for 16/32-blocks).
                # Accepts the full-width tensor or a GPTQ column slice vs a 1-block plan.
                return fmt_lib.fake_quant_ste_with_plan(
                    x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x,
                    self._fmt_plan).to(x_dtype).reshape(*init_shape)
            if self.groupsize != -1:
                x = x.reshape(-1, self.groupsize)
            if self.sym:
                x = sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            else:
                x = asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
            x = x.reshape(*init_shape)
        return x
    
    def forward(self, x):
        return self.quantize(x)

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def set_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, (WeightQuantizer, ActivationQuantizer)):
            m.enable = enable
    return model


def set_weight_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, WeightQuantizer):
            m.enable = enable
    return model


def set_act_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, ActivationQuantizer):
            m.enable = enable
    return model


# --- E1-0 format-selection freeze: layer-level observe orchestration --------------------

_KV_ROLE_SUFFIX = (("q_cache_quantizer", "q"), ("k_cache_quantizer", "k"),
                   ("v_cache_quantizer", "v"))


def _infer_role(relname, module):
    """Role of a quantizer from its module-tree name (no construction-site tagging needed)."""
    if isinstance(module, WeightQuantizer):
        return "w"
    for suffix, role in _KV_ROLE_SUFFIX:          # ActivationQuantizer reused for KV/Q cache
        if relname.endswith(suffix):
            return role
    return "a"                                     # qkv_quant / up_gate_quant / o,down act_quant


def start_observe_layer(layer, fmt_pools, layer_key_prefix):
    """Put every quantizer whose role has a candidate pool into observe mode. Site key =
    '<layer_key_prefix>.<module-tree name>' (matches the key gptq/rtn reconstruct for weights)."""
    n = 0
    for relname, m in layer.named_modules():       # named_modules dedups shared quantizers
        if not isinstance(m, (WeightQuantizer, ActivationQuantizer)):
            continue
        if getattr(m, "bits", 16) >= 16:
            continue
        role = _infer_role(relname, m)
        entry = fmt_pools.get(role)
        if entry is None:
            continue
        _observe_start(m, entry["pool"], role, unify_scale=entry.get("select", "mse"),
                       site_key=f"{layer_key_prefix}.{relname}")
        n += 1
    return n


def finalize_observe_layer(layer):
    """Finalize (argmin + freeze fmt_cfg) every observing quantizer; return selection records."""
    recs = []
    for _relname, m in layer.named_modules():
        if isinstance(m, (WeightQuantizer, ActivationQuantizer)) and getattr(m, "_observe", False):
            recs.append(_observe_finalize(m))
    return recs

