"""Why the standalone Triton ozaki GEMM is slower than cuBLAS on LARGE (compute-bound) GEMMs but
faster on the attention-shaped (low-intensity, batched) ones: a plain bf16 Triton GEMM (no ozaki),
swept over tiling, vs cuBLAS torch.bmm. Shows the gap is the small tile size (BM=BN=64, forced by the
ozaki digit-plane SRAM pressure), NOT the ozaki math -- 128x128 tiles match cuBLAS on large shapes,
while at low intensity even 64x64 Triton beats cuBLAS. Run from repo root:
    CUDA_VISIBLE_DEVICES=2 python flash_ozaki/verification/analyze_triton_vs_cublas.py"""
import sys, time, torch, triton, triton.language as tl
dev="cuda"; torch.manual_seed(0)

@triton.jit
def _bf16_gemm(A,B,C,M,N,K, saz,sam,sak, sbz,sbk,sbn, scz,scm,scn,
              CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1); pid_z=tl.program_id(2)
    offm=pid_m*BM+tl.arange(0,BM); offn=pid_n*BN+tl.arange(0,BN)
    mm=offm<M; nm=offn<N
    acc=tl.zeros([BM,BN],tl.float32)
    for kc in range(0,K,CHUNK):
        offk=kc+tl.arange(0,CHUNK); km=offk<K
        a=tl.load(A+pid_z*saz+offm[:,None]*sam+offk[None,:]*sak, mask=mm[:,None]&km[None,:], other=0.0)
        b=tl.load(B+pid_z*sbz+offk[:,None]*sbk+offn[None,:]*sbn, mask=km[:,None]&nm[None,:], other=0.0)
        acc+=tl.dot(a,b,out_dtype=tl.float32)
    tl.store(C+pid_z*scz+offm[:,None]*scm+offn[None,:]*scn, acc.to(tl.bfloat16), mask=mm[:,None]&nm[None,:])

def tri_gemm(A,B,BM,BN,CHUNK,ns=2,nw=4):
    Z,M,K=A.shape; N=B.shape[2]
    C=torch.empty(Z,M,N,device=dev,dtype=torch.bfloat16)
    grid=(triton.cdiv(M,BM),triton.cdiv(N,BN),Z)
    _bf16_gemm[grid](A,B,C,M,N,K, A.stride(0),A.stride(1),A.stride(2), B.stride(0),B.stride(1),B.stride(2),
                     C.stride(0),C.stride(1),C.stride(2), CHUNK=CHUNK,BM=BM,BN=BN,num_stages=ns,num_warps=nw)
    return C

def bench(fn,n=40,w=15):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3

def run(tag,Z,M,K,N):
    A=torch.randn(Z,M,K,device=dev,dtype=torch.bfloat16); B=torch.randn(Z,K,N,device=dev,dtype=torch.bfloat16)
    tc=bench(lambda: torch.bmm(A,B))
    print(f"\n[{tag}] Z={Z} M={M} K={K} N={N} | cuBLAS bmm = {tc:.3f} ms")
    for (BM,BN,CH,ns) in [(64,64,32,2),(64,64,64,3),(128,128,32,3),(128,128,64,3),(128,64,64,4)]:
        try:
            t=bench(lambda: tri_gemm(A,B,BM,BN,CH,ns))
            print(f"    Triton bf16 GEMM BM{BM} BN{BN} CHUNK{CH} ns{ns}: {t:.3f} ms  ({t/tc:.2f}x cuBLAS)")
        except Exception as e:
            print(f"    BM{BM} BN{BN} CHUNK{CH}: OOR/{str(e)[:30]}")

run("LARGE compute-bound", 1, 2048, 4096, 4096)
run("QK-small low-intensity", 28, 1024, 128, 1024)
print("\nDONE")
