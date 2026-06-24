"""Measure vLLM generation throughput on the same 5 MATH-500 problems / settings as the
Ozaki nmp=1 run, for a speed comparison. bf16 DeepSeek-R1-Distill-Qwen-7B, vLLM backend,
same generation params as inference_vllm.py (temp 0.6, top_p 0.95, max 32768, enforce_eager)."""
import json
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"

tok = AutoTokenizer.from_pretrained(MODEL)
problems = [json.loads(l) for l in open("datasets/MATH-500/test.jsonl")][:5]
prompts = []
for p in problems:
    msg = [{"role": "user", "content": f"{p['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
    prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))

llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_model_len=32768,
          gpu_memory_utilization=0.9, enable_prefix_caching=False, enable_chunked_prefill=False)
sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768, seed=42)

# warm-up excluded: time only the batched generate of all 5 prompts (vLLM runs them in parallel)
t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0

out_tok = [len(o.outputs[0].token_ids) for o in outs]
total = sum(out_tok)
print("\n==== vLLM speed (5 MATH-500, bf16, enforce_eager) ====")
print(f"per-sample output tokens: {out_tok}")
print(f"total output tokens: {total}")
print(f"wall-clock (5 in parallel): {dt:.1f}s")
print(f"aggregate throughput: {total/dt:.1f} tok/s")
print(f"avg per-sample length: {total/len(outs):.0f} tokens")
