"""Part G: flash-exact (`_flash_exact_fwd`, the ozaki=False bf16 path) vs the **vllm-flash-attn CUDA
kernel** vLLM actually serves with, decomposed stage by stage against an fp64 reference -- plus the
verification of the `pv_split` fix that removes the decode-only deficit.

Part B compared flash-exact against torch SDPA only. That missed a real (if tiny) difference: at decode
our kernel sat ~4.7% farther from truth than vllm-flash-attn, growing with context. This script locates
that error, shows everything else (tiling, online softmax, GQA fold) is 4 orders of magnitude below it,
and verifies the fix does not disturb prefill, the ozaki path, or kv_lens masking.

  python flash_ozaki/verification/analyze_vs_vllm_flash_attn.py            # decomposition + fix
  python flash_ozaki/verification/analyze_vs_vllm_flash_attn.py --quick    # skip the 20-draw sweeps
"""
import argparse, math, os, sys
import torch, triton

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from flash_ozaki.flash_oz1fp_codegen import (flash_oz1fp_cg, flash_oz1fp_cg_splitkv, fa_num_splits,
                                             decode_num_splits)
from vllm.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
HQ, HKV, D = 14, 2, 128            # Qwen2-7B under TP=2: per-GPU query/kv heads, head_dim
SM = 1.0 / math.sqrt(D)


def fa_cuda(q, k, v, causal=True):
    """The exact entry points vLLM's FlashAttentionImpl uses: varlen for prefill, with_kvcache for
    decode. q:[B,Hq,T,D] k,v:[B,Hkv,N,D] -> [B,Hq,T,D]."""
    B, Hq, T, Dh = q.shape
    N = k.shape[2]
    qh, kh, vh = (t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v))
    if T == 1:
        o = flash_attn_with_kvcache(
            qh, kh, vh, cache_seqlens=torch.full((B,), N, dtype=torch.int32, device=q.device),
            softmax_scale=SM, causal=causal)
    else:
        cuq = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
        cuk = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=q.device)
        o = flash_attn_varlen_func(
            qh.reshape(B * T, Hq, Dh), kh.reshape(B * N, k.shape[1], Dh),
            vh.reshape(B * N, v.shape[1], Dh), cu_seqlens_q=cuq, cu_seqlens_k=cuk,
            max_seqlen_q=T, max_seqlen_k=N, softmax_scale=SM, causal=causal).reshape(B, T, Hq, Dh)
    return o.permute(0, 2, 1, 3)


def ref_fp64(q, k, v, causal=True):
    """Ground truth: fp64, full-row softmax, no tiling."""
    T, N, G = q.shape[2], k.shape[2], q.shape[1] // k.shape[1]
    kk, vv = (t.repeat_interleave(G, 1).double() for t in (k, v))
    s = (q.double() @ kk.transpose(-1, -2)) * SM
    if causal:
        qi = torch.arange(T, device=q.device) + (N - T)
        s = s.masked_fill(qi[:, None] < torch.arange(N, device=q.device)[None, :], float("-inf"))
    return torch.softmax(s, -1) @ vv


def sim(q, k, v, block_n, p_bf16, out_bf16, dt=torch.float32, causal=True):
    """Torch twin of a flash datapath: tiled online softmax at width block_n, optional P->bf16 before
    P@V, optional bf16 output store. dt selects the accumulator precision."""
    B, Hq, T, _ = q.shape
    N, G = k.shape[2], Hq // k.shape[1]
    kk, vv = (t.repeat_interleave(G, 1).to(dt) for t in (k, v))
    qq = q.to(dt)
    qi = torch.arange(T, device=q.device) + (N - T)
    m_i = torch.full((B, Hq, T), -float("inf"), device=q.device, dtype=dt)
    l_i = torch.zeros((B, Hq, T), device=q.device, dtype=dt)
    acc = torch.zeros((B, Hq, T, D), device=q.device, dtype=dt)
    for n0 in range(0, N, block_n):
        n1 = min(n0 + block_n, N)
        s = (qq @ kk[:, :, n0:n1].transpose(-1, -2)) * SM
        if causal:
            s = s.masked_fill(qi[:, None] < torch.arange(n0, n1, device=q.device)[None, :], -float("inf"))
        m_new = torch.maximum(m_i, s.max(-1).values)
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(s - m_new[..., None])
        l_i = l_i * alpha + p.sum(-1)
        acc = acc * alpha[..., None] + (p.to(torch.bfloat16).to(dt) if p_bf16 else p) @ vv[:, :, n0:n1]
        m_i = m_new
    o = acc / l_i[..., None]
    return o.to(torch.bfloat16).to(dt) if out_bf16 else o


def sim_pbits(q, k, v, bits, block_n=32):
    """sim() with P rounded to `bits` significant bits instead of bf16 -- probes which precision class
    a black-box kernel's P sits in (bits=8 reproduces bf16)."""
    B, Hq, T, _ = q.shape
    N, G = k.shape[2], Hq // k.shape[1]
    kk, vv = (t.repeat_interleave(G, 1).float() for t in (k, v))
    qq = q.float()
    qi = torch.arange(T, device=q.device) + (N - T)
    m_i = torch.full((B, Hq, T), -float("inf"), device=q.device)
    l_i = torch.zeros((B, Hq, T), device=q.device)
    acc = torch.zeros((B, Hq, T, D), device=q.device)
    for n0 in range(0, N, block_n):
        n1 = min(n0 + block_n, N)
        s = (qq @ kk[:, :, n0:n1].transpose(-1, -2)) * SM
        s = s.masked_fill(qi[:, None] < torch.arange(n0, n1, device=q.device)[None, :], -float("inf"))
        m_new = torch.maximum(m_i, s.max(-1).values)
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(s - m_new[..., None])
        l_i = l_i * alpha + p.sum(-1)
        mm, ee = torch.frexp(p)                          # round to `bits` significant bits
        acc = acc * alpha[..., None] + torch.ldexp(torch.round(mm * 2 ** bits) / 2 ** bits, ee) @ vv[:, :, n0:n1]
        m_i = m_new
    return (acc / l_i[..., None]).to(torch.bfloat16).float()


relerr = lambda x, ref: ((x.double() - ref).norm() / ref.norm()).item()
bias = lambda x, ref: ((x.double() - ref).mean() / ref.abs().mean()).item()
ours = lambda q, k, v, **kw: flash_oz1fp_cg(q, k, v, nmp=1, w=8, causal=True, sm_scale=SM,
                                            chunk_size=32, ozaki=False, **kw)


def qkv(B, T, N, seed, heavy=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, HQ, T, D, generator=g, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, HKV, N, D, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, HKV, N, D, generator=g, device=DEV, dtype=torch.bfloat16)
    if heavy:                       # outlier channels + attention sink, as in real LLM activations
        q[..., :4] *= 20.0; k[..., :4] *= 20.0; k[:, :, 0, :] *= 8.0
    return q, k, v


def main(quick):
    q, k, v = qkv(8, 1, 2048, 0)
    ref = ref_fp64(q, k, v)
    print("=" * 96)
    print("G.1  stage decomposition (DECODE B=8 N=2048, relerr vs fp64)")
    print("=" * 96)
    for lab, kw in [("fp64 tiled online softmax (tile=32)", dict(block_n=32, p_bf16=False, out_bf16=False, dt=torch.float64)),
                    ("+ fp32 accumulators",                 dict(block_n=32, p_bf16=False, out_bf16=False)),
                    ("+ P -> bf16 before P@V",              dict(block_n=32, p_bf16=True,  out_bf16=False)),
                    ("+ bf16 output store [= old kernel]",  dict(block_n=32, p_bf16=True,  out_bf16=True)),
                    ("same at tile=128 [= FA-like]",        dict(block_n=128, p_bf16=True, out_bf16=True))]:
        print(f"  {lab:<40}{relerr(sim(q, k, v, **kw), ref):>12.2e}")
    e32 = sim(q, k, v, 32, False, False); e128 = sim(q, k, v, 128, False, False)
    print(f"  {'tile 32 vs 128, no P/out cast':<40}{relerr(e32, e128.double()):>12.2e}   <-- tiling is a non-issue")

    print("\n" + "=" * 96)
    print("G.2/G.3  distance to fp64: old (single bf16 dot) vs new (hi/lo split) vs vllm-flash-attn")
    print("=" * 96)
    shapes = [("PREFILL T=N=92", 8, 92, 92), ("PREFILL T=N=512", 4, 512, 512),
              ("DECODE  T=1 N=512", 8, 1, 512), ("DECODE  T=1 N=2048", 8, 1, 2048),
              ("DECODE  T=1 N=4368", 4, 1, 4368), ("DECODE  T=1 N=16384", 2, 1, 16384)]
    draws = 3 if quick else 20
    print(f"{'shape':<22}{'old':>12}{'new':>12}{'vllm-FA':>12}{'new/FA':>9}{'new worse':>11}")
    for lab, B, T, N in shapes:
        eo, en, ef, worse = [], [], [], 0
        for s in range(draws):
            qq, kk, vv = qkv(B, T, N, 500 + s)
            r = ref_fp64(qq, kk, vv)
            a, b, c = (relerr(ours(qq, kk, vv, pv_split=False), r),
                       relerr(ours(qq, kk, vv, pv_split=(T == 1)), r),
                       relerr(fa_cuda(qq, kk, vv), r))
            eo.append(a); en.append(b); ef.append(c); worse += (b > c)
        mo, mn, mf = (sum(x) / draws for x in (eo, en, ef))
        print(f"{lab:<22}{mo:>12.3e}{mn:>12.3e}{mf:>12.3e}{mn/mf:>9.3f}{worse:>8}/{draws}")
    print("  (prefill defaults to unsplit -- 'old' and 'new' are the same kernel launch there)")

    print("\n" + "=" * 96)
    print("G.2b  how many significant bits does each path carry P with? (DECODE B=8 N=2048)")
    print("=" * 96)
    e_fa, e_old = relerr(fa_cuda(q, k, v), ref), relerr(ours(q, k, v, pv_split=False), ref)
    for b in range(7, 17):
        e = relerr(sim_pbits(q, k, v, b), ref)
        print(f"  P at {b:>2} bits{e:>14.3e}" + ("   <== same class as vllm-FA" if abs(e - e_fa) / e_fa < 0.04 else ""))
    print(f"\n  {'ours OLD (one bf16 dot)':<28}{e_old:>12.3e}")
    print(f"  {'vllm-FA':<28}{e_fa:>12.3e}   <- also an 8-bit-P path; the ~3% gap is dot/reduction order")
    print(f"  {'ours NEW (hi/lo ~16 bits)':<28}{relerr(ours(q, k, v), ref):>12.3e}")
    print(f"  {'floor: P exact, bf16 out':<28}{relerr(sim(q, k, v, 32, False, True), ref):>12.3e}"
          "   <- the new kernel sits AT this floor")

    print("\n" + "=" * 96)
    print("G.2c  FA's decode edge is split-KV: pin its num_splits and the gap closes")
    print("=" * 96)
    B0, N0 = 8, 2048
    q0, k0, v0 = qkv(B0, 1, N0, 0)
    r0 = ref_fp64(q0, k0, v0)
    qh, kh, vh = (t.permute(0, 2, 1, 3).contiguous() for t in (q0, k0, v0))
    cs_ = torch.full((B0,), N0, dtype=torch.int32, device=q0.device)
    for ns in (1, 2, 4, 8, 16, 0):
        o = flash_attn_with_kvcache(qh, kh, vh, cache_seqlens=cs_, softmax_scale=SM, causal=True,
                                    num_splits=ns).permute(0, 2, 1, 3)
        print(f"  vllm-FA num_splits={'auto' if ns == 0 else ns:<5} {relerr(o, r0):.4e}"
              + ("   <== matches our single-pass path" if ns == 1 else ""))
    print(f"  ours single-pass{'':<13} {relerr(ours(q0, k0, v0, pv_split=False), r0):.4e}")

    print("\n" + "=" * 96)
    print("G.3  exact split-KV (FA's structure + FA-sized split count) vs vllm-FA")
    print("=" * 96)
    print(f"{'shape':<18}{'splits':>7}{'single-pass':>13}{'split-KV':>12}{'vllm-FA':>12}{'ratio':>8}")
    for B0, N0 in [(2, 512), (8, 2048), (32, 2048), (4, 4368), (16, 4368)]:
        q0, k0, v0 = qkv(B0, 1, N0, 0)
        r0 = ref_fp64(q0, k0, v0)
        zc, nt = B0 * HKV, max(1, -(-N0 // 32))
        ns = fa_num_splits(zc, nt)
        o_s = (ours(q0, k0, v0, pv_split=False) if ns <= 1 else
               flash_oz1fp_cg_splitkv(q0, k0, v0, nmp=1, w=8, causal=True, sm_scale=SM, chunk_size=32,
                                      n_splits=ns, ozaki=False))
        e1, es, ef = (relerr(ours(q0, k0, v0, pv_split=False), r0), relerr(o_s, r0),
                      relerr(fa_cuda(q0, k0, v0), r0))
        print(f"{f'B={B0} N={N0}':<18}{ns:>7}{e1:>13.3e}{es:>12.3e}{ef:>12.3e}{es/ef:>8.3f}")
    q0, k0, v0 = qkv(4, 1, 2048, 1)
    same = torch.equal(ours(q0, k0, v0, pv_split=False),
                       flash_oz1fp_cg_splitkv(q0, k0, v0, nmp=1, w=8, causal=True, sm_scale=SM,
                                              chunk_size=32, n_splits=1, ozaki=False))
    print(f"  n_splits=1 bit-identical to the single-pass kernel: {same}")

    print("\n" + "=" * 96)
    print("G.4  bitwise agreement of the bf16 output with vllm-FA (100% is unreachable, see RESULTS)")
    print("=" * 96)
    fa_o = fa_cuda(q, k, v)
    for lab, x in [("ours OLD", ours(q, k, v, pv_split=False)), ("ours NEW", ours(q, k, v))]:
        same = (x.view(torch.int16) == fa_o.view(torch.int16)).float().mean().item() * 100
        print(f"  {lab:<12} identical output elements {same:5.1f}%")
    print("  (more accurate == MORE agreement: both converge on the same true value)")

    print("\n" + "=" * 96)
    print("G.3  signed bias (mean(err)/mean|ref|) -- rounding does not bias, a defect would")
    print("=" * 96)
    for lab, B, T, N in shapes[:1] + shapes[3:4]:
        qq, kk, vv = qkv(B, T, N, 0)
        r = ref_fp64(qq, kk, vv)
        print(f"  {lab:<22} ours {bias(ours(qq, kk, vv), r):>10.2e}   FA {bias(fa_cuda(qq, kk, vv), r):>10.2e}")
    qq, kk, vv = qkv(4, 1, 2048, 7, heavy=True)
    r = ref_fp64(qq, kk, vv)
    print(f"  {'heavy-tailed inputs':<22} ours {relerr(ours(qq,kk,vv), r):>10.2e}   FA {relerr(fa_cuda(qq,kk,vv), r):>10.2e}")

    print("\n" + "=" * 96)
    print("G.3  regressions: prefill bit-identical / ozaki untouched / kv_lens mask still exact")
    print("=" * 96)
    for B, T in [(8, 92), (4, 512)]:
        qq, kk, vv = qkv(B, T, T, 11)
        print(f"  prefill B={B} T={T:<5} default == unsplit: {torch.equal(ours(qq,kk,vv), ours(qq,kk,vv,pv_split=False))}")
    qq, kk, vv = qkv(2, 1, 1024, 3)
    for nmp in (1, 4, 10, 15):
        a = flash_oz1fp_cg(qq, kk, vv, nmp=nmp, w=4, causal=True, sm_scale=SM, chunk_size=32, ozaki=True)
        b = flash_oz1fp_cg(qq, kk, vv, nmp=nmp, w=4, causal=True, sm_scale=SM, chunk_size=32, ozaki=True,
                           pv_split=True)          # must be IGNORED on the ozaki path
        print(f"  ozaki w4 nmp={nmp:<3} identical with pv_split True/False: {torch.equal(a, b)}")
    lens = torch.tensor([2048, 1500, 733, 91, 2048, 12, 999, 1777], device=DEV)
    qq, kp, vp = qkv(8, 1, 2048, 0)
    kj, vj = kp.clone(), vp.clone()
    for b in range(8):              # garbage tail x50, as the clamped paged-cache gather produces
        kj[b, :, lens[b]:] = torch.randn_like(kj[b, :, lens[b]:]) * 50
        vj[b, :, lens[b]:] = torch.randn_like(vj[b, :, lens[b]:]) * 50
    op = ours(qq, kj, vj, kv_lens=lens)
    worst = max((op[b:b+1].double() - ours(qq[b:b+1], kp[b:b+1, :, :int(lens[b])],
                                           vp[b:b+1, :, :int(lens[b])]).double()).abs().max().item()
                for b in range(8))
    print(f"  kv_lens padded vs per-seq exact: worst {worst:.3e}  ({'BIT-EXACT' if worst == 0 else 'LEAK'})")

    if not quick:
        print("\n" + "=" * 96)
        print("G.3  latency cost of the second dot (decode, ms/call)")
        print("=" * 96)
        for B, N in [(8, 2048), (32, 2048), (8, 4368), (64, 4368)]:
            qq, kk, vv = qkv(B, 1, N, 1)
            t0 = triton.testing.do_bench(lambda: ours(qq, kk, vv, pv_split=False))
            t1 = triton.testing.do_bench(lambda: ours(qq, kk, vv, pv_split=True))
            print(f"  B={B:<3} N={N:<6} unsplit {t0:.4f}  split {t1:.4f}   overhead {100*(t1/t0-1):+6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 draws instead of 20, skip the latency bench")
    main(ap.parse_args().quick)
