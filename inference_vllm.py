import os
import json
import random
import hashlib
import argparse
from tqdm import tqdm

import torch
import transformers
from vllm import LLM
from vllm.engine.arg_utils import PoolerConfig

from lighteval.models.model_input import GenerationParameters
from lighteval_custom.models.vllm.vllm_model import VLLMModelConfig
from lighteval_custom.main_vllm import vllm
from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
register_fake_quantized_models()    # register fake-quantized models in vLLM


def _make_run_tag(args):
    """Output-dir name for a run: legible ``key=value`` parts + an 8-char md5 of the full
    config. Mirrors emulation/llm/utils.py::make_result_filename so runs are self-describing
    and any two distinct configs land in distinct dirs. The dataset is the file name
    (<dataset>.jsonl) INSIDE the dir, not part of the tag, so one config dir collects every
    dataset run with that config. Perf-only knobs (gpu mem util, batch caps) and run-control
    flags are excluded from the hash since they don't change the results.
    """
    if args.ozaki_placement is None:
        keys = ["model", "dtype", "seed"]
    elif args.rslt_type in ("ozaki1", "ozaki1_fp"):
        keys = ["model", "ozaki_placement", "rslt_type", "nmp", "k", "weight_cache", "dtype", "seed"]
    else:
        keys = ["model", "ozaki_placement", "rslt_type", "s", "k", "scale_method",
                "shift_bits", "M_frac_bits", "weight_cache", "combine_fp64", "dtype", "seed"]
    parts = []
    for k in keys:
        v = getattr(args, k)
        if isinstance(v, bool):
            v = int(v)
        elif "/" in str(v):
            v = str(v).rstrip("/").split("/")[-1]
        parts.append(f"{k}={v}")
    # Hash the config that actually affects THIS run. Always drop run-control / derived / perf
    # fields; additionally drop Ozaki params that don't apply (all of them when ozaki is off,
    # the other scheme's params when on) so numerically-identical runs share one dir.
    exclude = {"config", "output_dir", "output_path", "model_name", "tensor_parallel_size",
               "overwrite", "debug", "dataset", "gpu_memory_utilization",
               "max_num_batched_tokens", "max_num_seqs"}
    ozaki1_only = {"nmp", "nmp_overrides", "gemm_bits", "byte_split_style"}
    ozaki2_only = {"s", "scale_method", "shift_bits", "M_frac_bits", "combine_fp64", "s_overrides"}
    if args.ozaki_placement is None:
        exclude |= {"ozaki_placement", "rslt_type", "k", "weight_cache", "ozaki_arch"} | ozaki1_only | ozaki2_only
    elif args.rslt_type in ("ozaki1", "ozaki1_fp"):
        exclude |= ozaki2_only
    else:
        exclude |= ozaki1_only
    cfg = {k: v for k, v in vars(args).items() if k not in exclude}
    h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return "__".join(parts) + f"__{h}"


def parser_gen():
    # _kv_int_map parses a per-op override map ("pat=val,..." on the CLI or a YAML {pat: val}
    # dict) identically to evaluate_ppl.py; needed as the `type=` for --nmp_overrides/--s_overrides
    # below, so it must be imported before those add_argument calls.
    from emulation.llm.config import _kv_int_map
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help="Path to a YAML config file whose entries seed the argparse defaults; "
                             "explicit CLI flags still win (precedence: CLI > YAML > default). "
                             "Mirrors emulation/llm/evaluate_ppl.py. YAML keys are the long flag "
                             "names without '--' (e.g. rslt_type, nmp, k, s, ozaki_placement).")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="whether to re-evaluate")
    parser.add_argument('--load_responses_from_json_file', type=str, default=None,
                        help='Load response from json file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to save inference results.')
    # model
    parser.add_argument('--model', type=str, default='./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B',
                        help='Model to load.')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype to use')
    # dataset
    parser.add_argument('--dataset', type=str, default='AIME-2024',
                        choices=["AIME-2024", "AIME-2025", "AIME-90", "MATH-500", "NuminaMath-1.5", "GSM8K", "GPQA-Diamond", "LiveCodeBench", "PPL"],
                        help='Dataset to load.')
    parser.add_argument('--max_samples', type=int, default=None, help='Max #samples (for debug)')
    # generation
    parser.add_argument('--temperature', type=float, default=0.6, help='Generation temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Generation top_p')
    parser.add_argument('--seed', type=int, default=42, help='Generation seed')
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='Maximum number of tokens to generate per output sequence.')
    parser.add_argument('--max_model_length', type=int, default=32768,
                        help='Maximum model input length.')
    # === Ozaki emulation on vLLM (continuous batching + paged KV) ===
    # --ozaki_placement chooses WHERE to emulate (vLLM-specific). The Ozaki GEMM params below
    # reuse the emulation's own flag names (emulation/llm/evaluate_ppl.py): rslt_type / nmp / k /
    # s / scale_method / shift_bits / M_frac_bits / weight_cache -- so a --config YAML shares the
    # same vocabulary as the emulation entry points.
    parser.add_argument('--ozaki_placement', type=str, default=None,
                        choices=['off', 'linear_only', 'full', 'attn_only'],
                        help='Where to apply Ozaki emulation (default / "off": no Ozaki -- plain '
                             'bf16/fp16 stock Qwen2). "off" is an explicit value so it can override '
                             'a --config YAML back to the baseline. '
                             'linear_only: nn.Linear only, attention stays native paged attention '
                             '(Qwen2OzakiForCausalLM; mirrors inference_transformers.py --linear_only). '
                             'full: linear AND attention QK^T/PV via the custom eager backend. '
                             'attn_only: attention QK^T/PV only, linear stays native bf16 (stock Qwen2).')
    parser.add_argument('--rslt_type', type=str, default='ozaki1_fp',
                        choices=['ozaki1', 'ozaki1_fp', 'ozaki2', 'ozaki2_fp'],
                        help='Ozaki GEMM type (emulation name). ozaki1 / ozaki1_fp: block-FP '
                             'polynomial (uses --nmp); ozaki2 / ozaki2_fp: RNS / CRT (uses --s). '
                             'A bare ozaki1/ozaki2 is the int8-decomposition GEMM; *_fp its bf16 '
                             'byte-plane emulation. ozaki2_fp is the validated path.')
    parser.add_argument('--nmp', type=int, default=6,
                        help='Ozaki-1 block-FP polynomial GEMM count (ozaki1 / ozaki1_fp).')
    parser.add_argument('--gemm_bits', type=int, default=8, choices=[2, 4, 8],
                        help='Ozaki-1 GEMM unit bit-width w (int_bits = w*nD-1). Default 8 = int8 '
                             'byte-split / native fold. w=2/4 use the w-bit paths and REQUIRE '
                             'rslt_type ozaki1 / ozaki1_fp; with ozaki1 (int8 reference) w!=8 has '
                             'no weight_cache, so set --weight_cache off. (ozaki2 ignores w.)')
    parser.add_argument('--byte_split_style', type=str, default='all_signed_clamp_pos',
                        choices=['all_signed_clamp_pos', 'all_signed_clamp', 'all_signed_no_clamp'],
                        help='Ozaki-1 integer digit-split (chunk) method: all_signed_clamp_pos '
                             '(default, canonical positive-only overflow prevention), '
                             'all_signed_clamp (legacy abs-max, conservative), all_signed_no_clamp '
                             '(MSB digit reaches +-2^(w-1); rslt_type ozaki1_fp ONLY). ozaki2 ignores it.')
    parser.add_argument('--k', type=int, default=32,
                        help='Ozaki reduction-dim chunk size (emulation --k).')
    # Ozaki-2 (RNS / CRT) params -- only used by ozaki2 / ozaki2_fp.
    parser.add_argument('--s', type=int, default=None,
                        help='Ozaki-2 RNS modulus count s (valid 2..20). REQUIRED for ozaki2 / ozaki2_fp.')
    parser.add_argument('--scale_method', type=str, default='new_compressed',
                        choices=['new_compressed', 'compressed', 'max'],
                        help='Ozaki-2 preprocessing scale method.')
    parser.add_argument('--shift_bits', type=int, default=7,
                        help='Ozaki-2 new_compressed scaling shift bits.')
    parser.add_argument('--M_frac_bits', type=int, default=8,
                        help='Ozaki-2 fractional bits of M in the scaling.')
    parser.add_argument('--weight_cache', action='store_true',
                        help='Enable Ozaki preprocessed-weight caching across forwards (faster but '
                             '~3x weight memory; emulation --weight_cache). Default off: re-encode '
                             'every forward, keeping only the 1x bf16 weight resident (single-GPU safe).')
    parser.add_argument('--combine_fp64', action='store_true',
                        help='Accumulate the ozaki2_fp place-value combine in fp64 (matches the '
                             'emulation ppl baseline, ~1e-7, ~2x slower). Default is fp32 '
                             'accumulation (faster; representative of real fixed/float accumulators), '
                             'exported as OZAKI2_COMBINE_FP64=0.')
    # Per-operation nmp/s overrides (emulation feature; same flag names as evaluate_ppl.py).
    # CLI string "qkv_proj=1,down_proj=3" or a YAML {pattern: value} map. Each pattern
    # re.search-matches the qualified linear-layer name; first match wins, else the default
    # --nmp / --s. NOTE: vLLM fuses q/k/v -> qkv_proj and gate/up -> gate_up_proj, so the
    # matchable names are qkv_proj / o_proj / gate_up_proj / down_proj (+ layers.<i>); the
    # transformers-side q_proj / k_proj names do not exist here. Only the linear-only / full
    # placements route linear layers, so overrides take effect there.
    parser.add_argument('--nmp_overrides', type=_kv_int_map, default=None,
                        help='Per-op nmp overrides for ozaki1 / ozaki1_fp; CLI "qkv_proj=1,o_proj=3" '
                             'or YAML {pattern: nmp}. Default nmp from --nmp applies elsewhere.')
    parser.add_argument('--s_overrides', type=_kv_int_map, default=None,
                        help='Per-op s overrides for ozaki2 / ozaki2_fp; CLI "qkv_proj=2,down_proj=4" '
                             'or YAML {pattern: s}. Default s from --s applies elsewhere.')
    parser.add_argument('--ozaki_arch', type=str, default='Qwen2OzakiForCausalLM',
                        help='Registered vLLM architecture to override into the model config.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_batched_tokens', type=int, default=None,
                        help='Cap per-step token batch. Bounds Ozaki intermediate memory; '
                             'defaults to 2048 for any Ozaki placement (chunked prefill enabled).')
    parser.add_argument('--max_num_seqs', type=int, default=None,
                        help='Cap concurrent sequences (bounds the decode-batch Ozaki memory).')
    # YAML base config + CLI override (CLI > YAML > default), mirroring evaluate_ppl.py.
    from emulation.llm.config import apply_yaml_config
    args = apply_yaml_config(parser)
    if args.ozaki_placement == "off":
        args.ozaki_placement = None  # "off" == no Ozaki (plain bf16/fp16); normalize to None

    # Ozaki-2 (RNS / CRT) needs an s value; the block-FP nmp does not apply to it.
    if args.ozaki_placement is not None and args.rslt_type in ("ozaki2", "ozaki2_fp"):
        if args.s is None:
            raise SystemExit(f"--rslt_type {args.rslt_type} (Ozaki-2 / RNS) requires "
                             "--s <int> (valid 2..20).")
        if not (2 <= args.s <= 20):
            raise SystemExit(f"--s must be in 2..20; got {args.s}.")
        # The ozaki2 Triton fold kernel does tl.arange(0, k), which Triton requires to be a
        # power of two. (Attention's smaller reduction dims are padded up to k at runtime.)
        if args.k & (args.k - 1) != 0 or args.k <= 0:
            raise SystemExit(f"--k must be a power of 2 for {args.rslt_type} (Triton fold "
                             f"kernel constraint); got {args.k}.")

    # gemm_bits (w) is an Ozaki-1 concept; w!=8 routes to the w-bit GEMM paths.
    if args.ozaki_placement is not None and args.gemm_bits != 8:
        if args.rslt_type not in ("ozaki1", "ozaki1_fp"):
            raise SystemExit(f"--gemm_bits {args.gemm_bits} (w!=8) only applies to rslt_type "
                             f"ozaki1 / ozaki1_fp; got {args.rslt_type}.")
        if args.rslt_type == "ozaki1" and args.weight_cache:
            raise SystemExit("--gemm_bits!=8 with rslt_type ozaki1 (int8 reference) has no "
                             "weight_cache; drop --weight_cache, or use rslt_type ozaki1_fp.")

    # byte_split_style (integer chunk method) is Ozaki-1 only; no_clamp needs the bf16 path.
    if args.ozaki_placement is not None and args.byte_split_style == "all_signed_no_clamp" \
            and args.rslt_type != "ozaki1_fp":
        raise SystemExit("--byte_split_style all_signed_no_clamp is rslt_type ozaki1_fp only "
                         "(the int8 path cannot represent the +-2^(w-1) MSB digit).")

    # force float16 for gptqmodel inference
    if "gptqmodel" in args.model:
        args.dtype = "float16"

    # output path: encode the run config (legible key=value parts) + an 8-char md5 of the full
    # config in the dir name (mirrors emulation/llm/utils.py::make_result_filename). One config
    # dir holds <dataset>.jsonl for each dataset run with it. --output_dir overrides.
    args.model_name = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join("./outputs", "inference", _make_run_tag(args))
    os.makedirs(output_dir, exist_ok=True)
    args.output_path = os.path.join(output_dir, f"{args.dataset}.jsonl")

    # Distributed settings
    args.tensor_parallel_size = torch.cuda.device_count()

    return args


class PPLEvaluator:
    def __init__(self, args):
        self.args = args
        self.max_length = 2048
        # LLM head
        llm_hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype)
        self.lm_head = llm_hf.lm_head
        self.lm_head.to("cuda:0")
        # LLM to output hidden_states before LLM head
        self.llm = LLM(model=args.model, dtype=args.dtype, enforce_eager=True,
                       tensor_parallel_size=args.tensor_parallel_size,
                       task="embed", override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False, softmax=False))

    @torch.no_grad()
    def __call__(self, testenc):
        print('Evaluating ppl...')
        testenc = testenc.input_ids
        nsamples = testenc.numel() // self.max_length

        nlls = []
        for i in tqdm(range(nsamples)):
            batch = {
                "prompt_token_ids": testenc[:, (i * self.max_length): ((i + 1) * self.max_length)].squeeze().tolist()
            }
            outputs = self.llm.encode(batch)
            hidden_states = outputs[0].outputs.data
            lm_logits = self.lm_head(hidden_states.to(self.lm_head.weight))
            shift_logits = lm_logits[:-1, :].contiguous()
            shift_labels = testenc[
                :, (i * self.max_length): ((i + 1) * self.max_length)
            ][:, 1:].to(shift_logits.device).squeeze()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.max_length
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * self.max_length))
        return ppl.item()


def main(args):
    if not args.debug and not args.overwrite and os.path.exists(args.output_path):
        print(f"Evaluation results found at {args.output_path}. Skip evaluation")
        return

    # Record the FULL resolved config (YAML + CLI + defaults, incl. derived fields) next to the
    # results. The dir-name hash isn't reversible, so this is the human-readable source of truth
    # for what produced <dataset>.jsonl.
    args_path = os.path.splitext(args.output_path)[0] + ".args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True, default=str)
    print(f"Run config saved at {args_path}.")

    random.seed(args.seed)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if args.debug:
        import debugpy
        debugpy.listen(5678)
        args.max_new_tokens = 10
        args.max_samples = 2

    generation_parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=30 if "QwQ" in args.model else None,  # TODO. enable top_k only for QwQ?
        max_new_tokens=args.max_new_tokens,
        seed=args.seed
    )
    ozaki_hf_overrides = None
    # Ozaki emulation expands activations into int8 chunk/plane tensors, so a full
    # max_model_len prefill batch can blow up GPU memory. Enable chunked prefill and cap
    # the per-step token batch to bound those intermediates (numerically equivalent).
    enable_chunked_prefill = False
    max_num_batched_tokens = args.max_num_batched_tokens
    max_num_seqs = args.max_num_seqs
    # Placement -> which paths are emulated. linear_ozaki swaps in the custom model
    # (Qwen2OzakiForCausalLM); use_ozaki_attn installs the custom attention backend.
    linear_ozaki = args.ozaki_placement in ("linear_only", "full")
    use_ozaki_attn = args.ozaki_placement in ("full", "attn_only")
    if linear_ozaki:
        # LINEAR ozaki -> custom model. attn_only deliberately does NOT set this, so linear
        # layers keep the stock (native bf16) path.
        ozaki_hf_overrides = {
            "architectures": [args.ozaki_arch],
            "ozaki_config": {
                "nmp": args.nmp,
                "chunk_size": args.k,
                "rslt_type": args.rslt_type,
                "weight_cache": args.weight_cache,
                # Ozaki-1 GEMM unit bit-width w (8 = int8 byte-split; 2/4 = w-bit). ozaki2 ignores it.
                "gemm_bits": args.gemm_bits,
                # Ozaki-1 integer digit-split (chunk) method. ozaki2 ignores it.
                "byte_split_style": args.byte_split_style,
                # Ozaki-2 (RNS) params; ignored by the ozaki1 types.
                "s": args.s,
                "scale_method": args.scale_method,
                "shift_bits": args.shift_bits,
                "M_frac_bits": args.M_frac_bits,
                # Per-op overrides: nmp_overrides for ozaki1*, s_overrides for ozaki2*
                # (the other is ignored by its scheme). {pattern: value} or None.
                "nmp_overrides": args.nmp_overrides,
                "s_overrides": args.s_overrides,
            },
        }
    if args.ozaki_placement is not None:
        enable_chunked_prefill = True
        if max_num_batched_tokens is None:
            max_num_batched_tokens = 2048
        if use_ozaki_attn and max_num_seqs is None:
            max_num_seqs = 48  # bound the decode pad-to-max attention gather at long context
        # ozaki2_fp place-value combine precision. DEFAULT fp32 accumulation (faster; closer to
        # real fixed/float accumulators); --combine_fp64 opts into the fp64 ppl baseline. Set
        # before the engine builds so spawned TP workers inherit it. Read by
        # emulation.ozaki2_batched_gemm_fp; only affects rslt_type ozaki2_fp.
        os.environ["OZAKI2_COMBINE_FP64"] = "1" if args.combine_fp64 else "0"

    model_config = VLLMModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        max_model_length=args.max_model_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=enable_chunked_prefill,
        generation_parameters=generation_parameters,
        init_model=(args.load_responses_from_json_file is None),
        hf_overrides=ozaki_hf_overrides,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    if args.dataset == "AIME-2024":
        task_kwargs = {
            "tasks": "custom|aime24|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "AIME-2025":
        task_kwargs = {
            "tasks": "custom|aime25|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "AIME-90":
        task_kwargs = {
            "tasks": "custom|aime90|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "MATH-500":
        task_kwargs = {
            "tasks": "custom|math_500|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "NuminaMath-1.5":
        task_kwargs = {
            "tasks": "custom|numina_math|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "GSM8K":
        task_kwargs = {
            "tasks": "custom|gsm8k|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "GPQA-Diamond":
        task_kwargs = {
            "tasks": "custom|gpqa:diamond|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "LiveCodeBench":
        task_kwargs = {
            "tasks": "custom|lcb:codegeneration|0|0",
            "custom_tasks": "lighteval_custom/tasks/livecodebench.py",
        }
    elif args.dataset == "PPL":
        from methods.utils import data_utils
        ppl_evaluator = PPLEvaluator(args)
        for eval_dataset in ["wikitext2"]:
        # for eval_dataset in ["wikitext2", "c4"]:
            print(eval_dataset)
            testloader = data_utils.get_loaders(
                eval_dataset,
                model=args.model,
                seqlen=2048,
                eval_mode=True
            )
            dataset_ppl = ppl_evaluator(testloader)
            print(dataset_ppl)
        return

    if use_ozaki_attn:
        # Install the custom ozaki attention backend BEFORE the LLM is built (attention
        # layers resolve their backend at construction). TP=1 runs in-process so patching
        # the driver suffices; TP>1 workers inherit OZAKI_* env vars set here.
        from vllm_custom.model_executor.layers.ozaki_attention import install_ozaki_attention_backend
        install_ozaki_attention_backend(args.nmp, args.k, args.rslt_type,
                                        s=args.s, scale_method=args.scale_method,
                                        shift_bits=args.shift_bits, M_frac_bits=args.M_frac_bits,
                                        gemm_bits=args.gemm_bits,
                                        byte_split_style=args.byte_split_style)

    results, details = vllm(
        model_config=model_config,
        use_chat_template=True,
        # output_dir="./outputs/lighteval_outputs",
        max_samples=args.max_samples,
        load_responses_from_json_file=args.load_responses_from_json_file,
        **task_kwargs,
    )

    # save evaluation results
    eval_results = []
    task_name = list(details.keys())[0]
    for detail in details[task_name]:
        eval_results.append({
            "full_prompt": detail["full_prompt"],
            "generated_text": detail["predictions"][0],
            "gold": detail["gold"],
            "metrics": detail["metrics"]
        })
    if not args.debug and args.load_responses_from_json_file is None:
        with open(args.output_path, "w") as f:
            json.dump(eval_results, f, indent=4)
        print(f"Evaluation results saved at {args.output_path}.")


if __name__ == "__main__":
    args = parser_gen()
    main(args)
