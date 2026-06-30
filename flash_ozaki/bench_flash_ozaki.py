"""Validate + benchmark the Triton Flash-Ozaki1_fp kernel (bf16 byte-plane) on a single GPU.

Validation: (1) exact mode vs torch exact attention (flash machinery); (2) ozaki mode vs EXACT
attention across (w,nmp) -> accuracy of fused ozaki1_fp attention.  (GEMM-level fidelity vs the
production batched_gemm is in oz1fp_triton.py.)
Speed: Flash-Ozaki1_fp vs Flash(exact, same kernel) vs torch SDPA(flash) vs materialized-eager.
"""
import sys, time
import torch

sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from flash_ozaki.flash_oz1fp_triton import flash_oz1fp
from flash_ozaki.oz1fp_triton import oz1fp_params


def relerr(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def exact_attn(q, k, v, causal=True):
    B, H, N, D = q.shape
    s = (q.float() @ k.float().transpose(-1, -2)) / (D ** 0.5)
    if causal:
        m = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~m, -float("inf"))
    return torch.softmax(s, -1) @ v.float()


def materialized_ozaki_eager(q, k, v, nmp, w, causal=True):
    """Non-fused ozaki1_fp attention via the production batched_gemm (the eager-backend math)."""
    from vllm_custom.model_executor.layers.ozaki_linear import build_ozaki_configs
    from emulation.llm.ozaki_matmul import batched_gemm
    import copy
    B, H, N, D = q.shape
    gcfg, oz = build_ozaki_configs(nmp=nmp, rslt_type="ozaki1_fp", chunk_size=32, weight_cache=False,
                                   gemm_bits=w, byte_split_style="all_signed_no_clamp")
    qf = q.reshape(B * H, N, D); kf = k.reshape(B * H, N, D); vf = v.reshape(B * H, N, D)
    cfg1 = copy.copy(gcfg); cfg1.name = "qk"
    s = batched_gemm(qf, kf.transpose(-1, -2), custom_gemm_config=cfg1, ozaki_config=oz,
                     out_dtype=torch.float32) / (D ** 0.5)
    if causal:
        m = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~m, -float("inf"))
    p = torch.softmax(s, -1)
    cfg2 = copy.copy(gcfg); cfg2.name = "pv"
    o = batched_gemm(p.to(torch.bfloat16), vf, custom_gemm_config=cfg2, ozaki_config=oz,
                     out_dtype=torch.float32)
    return o.reshape(B, H, N, D)


def do_bench(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    dev = "cuda"; torch.manual_seed(0)
    H, D, causal = 28, 128, True
    print(f"=== Flash-Ozaki1_fp validation (bf16 byte-plane; H={H}, D={D}, causal={causal}) ===")
    B, N = 1, 1024
    q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
    ex = exact_attn(q, k, v, causal)
    o_fp = flash_oz1fp(q, k, v, nmp=1, w=8, causal=causal, ozaki=False)
    print(f"  exact-mode kernel vs torch exact: rel-err = {relerr(o_fp, ex):.2e}  (flash machinery)")
    print(f"  {'config':>16} | {'fused vs EXACT':>14} | {'eager-ref vs EXACT':>18} | {'fused vs eager-ref':>18}")
    for (w, nmp) in [(8, 1), (4, 4), (4, 9), (4, 10), (4, 15)]:
        oz = flash_oz1fp(q, k, v, nmp=nmp, w=w, causal=causal, ozaki=True)
        try:
            eg = materialized_ozaki_eager(q, k, v, nmp=nmp, w=w, causal=causal)
            s2 = f"{relerr(eg,ex):18.2e} | {relerr(oz,eg):18.2e}"
        except Exception as e:
            s2 = f"{'(eager n/a: '+str(e)[:20]+')':>39}"
        nD = oz1fp_params(nmp, w)[0]
        print(f"  w={w} nmp={nmp:<2}(nD={nD}) | {relerr(oz,ex):14.2e} | {s2}")

    print(f"\n=== Speed (B={B}, H={H}, D={D}, causal); latency ms/call; ozaki=w4 nmp=10 ===")
    print(f"{'N_ctx':>6} | {'Flash-Ozaki1_fp':>15} | {'Flash exact(same krn)':>21} | {'torch SDPA':>11} | {'eager-ref(batched_gemm)':>23}")
    for N in [512, 1024, 2048, 4096]:
        q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
        t_oz = do_bench(lambda: flash_oz1fp(q, k, v, nmp=10, w=4, causal=causal, ozaki=True))
        t_fp = do_bench(lambda: flash_oz1fp(q, k, v, nmp=1, w=8, causal=causal, ozaki=False))
        t_sd = do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal))
        try:
            t_eg = do_bench(lambda: materialized_ozaki_eager(q, k, v, nmp=10, w=4, causal=causal), n=8, warmup=3)
            eg = f"{t_eg:23.2f}"
        except RuntimeError:
            eg = f"{'OOM':>23}"
        print(f"{N:6} | {t_oz:15.2f} | {t_fp:21.2f} | {t_sd:11.2f} | {eg}")


if __name__ == "__main__":
    main()
