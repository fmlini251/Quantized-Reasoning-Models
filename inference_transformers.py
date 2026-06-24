"""Lighteval (transformers backend) runner for the Ozaki nmp=1 DeepSeek-R1-Distill-Qwen-7B.

Mirrors inference_vllm.py (which uses the vLLM backend) but:
  * loads the bf16 Qwen2 model via lighteval's TransformersModel,
  * applies the Ozaki nmp=1 (rslt_type='ozaki1') int8-GEMM emulation to every Linear
    AND the attention score matmuls (QK^T / attn.V) via emulation.llm.ozaki_qwen,
  * keeps the same tasks (lighteval_custom/tasks/reasoning.py) and the same generation
    settings as the paper: temperature 0.6, top-p 0.95, max_new_tokens 32768, sampling on.

Run from the repo root (datasets are referenced as ./datasets/...), in the ozaki_ppl env:
    CUDA_VISIBLE_DEVICES=1 conda run -n ozaki_ppl python inference_transformers.py \
        --model ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B --dataset MATH-500 --seed 42
"""

import os
import json
import random
import argparse
from datetime import timedelta

import torch

# --- dataset -> lighteval task mapping (same as inference_vllm.py) ---
DATASET_TASKS = {
    "AIME-2024": ("custom|aime24|0|0", "lighteval_custom/tasks/reasoning.py"),
    "AIME-2025": ("custom|aime25|0|0", "lighteval_custom/tasks/reasoning.py"),
    "AIME-90": ("custom|aime90|0|0", "lighteval_custom/tasks/reasoning.py"),
    "MATH-500": ("custom|math_500|0|0", "lighteval_custom/tasks/reasoning.py"),
    "NuminaMath-1.5": ("custom|numina_math|0|0", "lighteval_custom/tasks/reasoning.py"),
    "GSM8K": ("custom|gsm8k|0|0", "lighteval_custom/tasks/reasoning.py"),
    "GPQA-Diamond": ("custom|gpqa:diamond|0|0", "lighteval_custom/tasks/reasoning.py"),
    "LiveCodeBench": ("custom|lcb:codegeneration|0|0", "lighteval_custom/tasks/livecodebench.py"),
}


def resolve_attn_impl(rslt_type, linear_only, requested):
    """Pick the concrete attention kernel ('eager' / 'sdpa') for the native attention path.

    Full Ozaki attention (Ozaki on the QK^T / attn.V matmuls) REQUIRES eager: the custom
    forward consumes the additive 4D causal mask that only the eager mask path emits (sdpa
    may hand it None). The bf16 baseline and --linear_only run the native Qwen2Attention,
    so either kernel is valid (both are full precision) -- 'auto' picks eager for bf16
    (matches the prior default) and sdpa for --linear_only.
    """
    full_ozaki_attn = (rslt_type != "bf16") and (not linear_only)
    if full_ozaki_attn:
        if requested == "sdpa":
            print("[warn] full-Ozaki attention requires eager (sdpa cannot host the int8 "
                  "QK^T/attn.V hook); ignoring --attn_impl sdpa.")
        return "eager"
    if requested != "auto":
        return requested
    return "sdpa" if linear_only else "eager"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--dataset", type=str, default="MATH-500", choices=list(DATASET_TASKS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nmp", type=int, default=1, choices=[1, 2, 3, 4, 6], help="ozaki1 nmp (int8 GEMM products)")
    p.add_argument("--rslt_type", type=str, default="ozaki1", choices=["ozaki1", "ozaki1_fp", "bf16"], help="ozaki matmul result type; 'bf16' = plain eager baseline (no ozaki)")
    p.add_argument("--gemm_bits", "--w", type=int, default=8, choices=[2, 4, 8], dest="gemm_bits",
                   help="ozaki1 GEMM unit bit-width w (int_bits=w*nD-1). 8=int8 byte-split / native "
                        "fold; 2/4=w-bit paths (ozaki1 -> int8 reference w/o weight_cache, ozaki1_fp "
                        "-> packed bf16). Ignored for bf16.")
    p.add_argument("--byte_split_style", type=str, default="all_signed_clamp_pos",
                   choices=["all_signed_clamp_pos", "all_signed_clamp", "all_signed_no_clamp"],
                   help="ozaki1 integer digit-split (chunk) method: all_signed_clamp_pos (default, "
                        "positive-only overflow prevention), all_signed_clamp (legacy abs-max), "
                        "all_signed_no_clamp (MSB digit +-2^(w-1); rslt_type ozaki1_fp ONLY).")
    p.add_argument("--chunk_size", type=int, default=128, help="reduction-dim chunk size (ozaki k)")
    p.add_argument("--linear_only", action="store_true",
                   help="apply Ozaki only to nn.Linear (q/k/v/o_proj, mlp, lm_head); keep the "
                        "attention score matmuls (QK^T/softmax/attn.V) in full precision via the "
                        "native attention kernel (see --attn_impl; SDPA by default)")
    p.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "eager", "sdpa"],
                   help="attention kernel for the NATIVE (non-Ozaki) attention path. 'auto': eager "
                        "for bf16, sdpa for --linear_only. Use this to run the bf16 baseline through "
                        "sdpa. Full-Ozaki attention always uses eager regardless (sdpa cannot host "
                        "the int8 QK^T/attn.V hook); a requested sdpa there is warned and ignored.")
    p.add_argument("--no_sanitize_logits", action="store_true", help="disable lm_head NaN/Inf logit sanitization")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=32768)
    p.add_argument("--max_model_length", type=int, default=32768)
    p.add_argument("--max_samples", type=int, default=None, help="limit #samples (subset / debug)")
    p.add_argument("--batch_size", type=int, default=1, help="generation batch size")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--shard_id", type=int, default=0, help="this shard's index in [0, num_shards)")
    p.add_argument("--num_shards", type=int, default=1, help="total shards for data-parallel split across GPUs")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.model_name = args.model.rstrip("/").split("/")[-1]
    args.attn_impl = resolve_attn_impl(args.rslt_type, args.linear_only, args.attn_impl)
    if args.output_dir is None:
        if args.rslt_type == "bf16":
            tag = f"inference_bf16_{args.attn_impl}"  # bf16_eager (default) or bf16_sdpa
        else:
            tag = f"inference_{args.rslt_type}_k{args.chunk_size}_nmp{args.nmp}"
            if args.linear_only:
                tag += "_linonly"
        args.output_dir = os.path.join("./outputs", tag, f"{args.model_name}-seed{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.num_shards > 1:
        args.output_path = os.path.join(args.output_dir, f"{args.dataset}.shard{args.shard_id}of{args.num_shards}.jsonl")
    else:
        args.output_path = os.path.join(args.output_dir, f"{args.dataset}.jsonl")
    return args


def build_ozaki_configs(nmp, rslt_type="ozaki1", chunk_size=128, gemm_bits=8,
                        byte_split_style="all_signed_clamp_pos"):
    """CustomGemmConfig + GlobalOzaki1Config for a single ozaki1/ozaki1_fp GEMM with the
    given nmp, matching evaluate_ppl.py's --rslt_type <rslt_type> --nmp <nmp> --k <chunk_size>."""
    # emulation refactor (~2026-06): GlobalOzakiConfig -> GlobalOzaki1Config; the ozaki1
    # config dropped the RNS / scale_method / shift_bits kwargs (ozaki1 never used them, so
    # prepare_ozaki([]) is no longer needed either).
    from emulation.llm.ozaki_matmul import CustomGemmConfig, GlobalOzaki1Config

    gcfg = CustomGemmConfig(
        in_feature_ts=4096, out_feature_ts=4096, chunk_size=chunk_size, name="",
        track_mtx_acc=False, track_model_acc=False, get_statistics=False, rslt_type=rslt_type,
    )
    # gemm_bits (=w): 8 = int8 byte-split / native fold; 2/4 = w-bit paths. The int8 reference
    # (rslt_type ozaki1) used for w!=8 has NO weight_cache, so disable it there; ozaki1_fp caches
    # at any w.
    use_weight_cache = not (rslt_type == "ozaki1" and gemm_bits != 8)
    oz = GlobalOzaki1Config(
        nmp_lst=[nmp], rounding="round_half_away_from_0",
        weight_cache=use_weight_cache,  # cache preprocessed weights across decode steps
        gemm_bits=gemm_bits,
        byte_split_style=byte_split_style,
    )
    return gcfg, oz


def make_ozaki_model_class(nmp, rslt_type="ozaki1", chunk_size=128, sanitize_logits=True,
                           linear_only=False, attn_impl="eager", gemm_bits=8,
                           byte_split_style="all_signed_clamp_pos"):
    from lighteval.models.transformers.transformers_model import TransformersModel
    from emulation.llm.ozaki_qwen import (
        prepare_qwen2_for_custom_matmul,
        replace_all_linears_with_custom_qwen,
    )

    class OzakiTransformersModel(TransformersModel):
        def _create_auto_model(self, config, env_config):
            model = super()._create_auto_model(config, env_config)
            # attn_impl is resolved upstream (resolve_attn_impl): always 'eager' for the
            # full-Ozaki attention path; the honored choice for bf16 / --linear_only.
            # Qwen2Attention reads config._attn_implementation at forward time, so setting
            # it post-load is enough to switch the native kernel (and the mask the model
            # emits).
            model.config._attn_implementation = attn_impl
            model.config.attn_implementation = attn_impl
            if rslt_type == "bf16":
                return model  # plain bf16 baseline (no Ozaki emulation)

            gcfg, oz = build_ozaki_configs(nmp, rslt_type, chunk_size, gemm_bits, byte_split_style)
            if linear_only:
                # Ozaki on every nn.Linear (q/k/v/o_proj, gate/up/down_proj, lm_head)
                # ONLY; the attention score matmuls (QK^T / softmax / attn.V) stay in full
                # precision via the native Qwen2Attention path (attn_impl above). Do NOT
                # swap the attention module.
                replace_all_linears_with_custom_qwen(model, gcfg, oz)
            else:
                # Full Ozaki attention: emulate QK^T / attn.V too. attn_impl is forced to
                # 'eager' upstream so the model produces the additive 4D causal mask that
                # custom_qwen2_eager_attention_forward consumes (sdpa may return None).
                prepare_qwen2_for_custom_matmul(model, gcfg, oz)
            self._ozaki_oz = oz  # keep a reference so buffers/config are not GC'd

            # The fp/int8 emulation can rarely emit NaN/Inf logits deep into a long
            # generation, which crashes torch.multinomial during sampling. Sanitize the
            # lm_head output so a bad step degrades to a finite (garbage) pick instead of
            # crashing the whole run (mirrors how production inference engines handle it).
            if sanitize_logits:
                _lm = model.lm_head
                _orig_lm_forward = _lm.forward

                def _safe_lm_forward(x, _f=_orig_lm_forward):
                    return torch.nan_to_num(_f(x), nan=0.0, posinf=3e4, neginf=-3e4)

                _lm.forward = _safe_lm_forward
            return model

        def __init__(self, env_config, config):
            super().__init__(env_config, config)
            # to_transformers_dict() hardcodes output_scores=True; for long
            # generations that stores a [steps x vocab] tensor and OOMs. greedy_until
            # only needs sequences, so disable it.
            self.generation_config_dict["output_scores"] = False
            # Cap on generated tokens (set externally). The paper uses 32768, which is
            # infeasible through the Python int8 emulation (~seconds/token); keep this
            # small for real runs.
            self.generation_cap = None

        def _generate(self, *args, **kwargs):
            # Paper setting is sampling (temperature 0.6 / top-p 0.95). These reasoning
            # tasks otherwise request greedy (do_sample=False), which would ignore the
            # sampling params, so force sampling here to match the vLLM path.
            kwargs["do_sample"] = True
            return super()._generate(*args, **kwargs)

        def greedy_until(self, requests, override_bs=None):
            # Override of TransformersModel.greedy_until with two fixes for emulation:
            #   1) padding="longest" instead of "max_length": the base method pads every
            #      prompt up to min(ctx+generation_size, max_length). With generation_size
            #      =32768 that pads short prompts to tens of thousands of tokens, which
            #      makes attention run over all that padding (catastrophic for the int8
            #      emulation) and clamps max_new_tokens to 1 when max_length is small.
            #   2) cap max_new_tokens at self.generation_cap.
            import lighteval.models.transformers.transformers_model as _tm

            for request in requests:
                request.stop_sequence = _tm.as_list(request.stop_sequence) + [self.tokenizer.eos_token]
                request.tokenized_context = self.tok_encode(request.context)

            dataset = _tm.GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
            starting_batch_size = _tm.STARTING_BATCH_SIZE
            results = []

            for split_start, split_end in _tm.tqdm(
                dataset.splits_start_end_iterator(), total=dataset.num_dataset_splits,
                desc="Splits", position=0, disable=self.disable_tqdm,
            ):
                if dataset[0].generation_size is None:
                    max_context_continuation_size_allowed = self.max_length
                else:
                    longest = len(dataset[0].tokenized_context) + dataset[0].generation_size
                    max_context_continuation_size_allowed = min(longest, self.max_length)
                batch_size = self._get_batch_size(
                    override_bs=override_bs, max_input_length=max_context_continuation_size_allowed,
                    starting_batch_size=starting_batch_size,
                )
                starting_batch_size = batch_size * 2

                dataloader = _tm.DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: b)
                if self.accelerator:
                    dataloader = self.accelerator.prepare(dataloader)

                for batch in _tm.tqdm(
                    dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm,
                ):
                    stop_tokens = [] if self.use_chat_template else batch[0].stop_sequence
                    max_new_tokens = batch[0].generation_size
                    returns_logits = batch[0].use_logits
                    num_samples = batch[0].num_samples
                    do_sample = batch[0].do_sample
                    context = [c.context for c in batch]

                    tokenized = self.tokenizer(
                        context,
                        truncation="longest_first",
                        padding="longest",  # FIX 1: dynamic padding
                        return_tensors="pt",
                        max_length=max_context_continuation_size_allowed,
                        add_special_tokens=self.add_special_tokens,
                    ).to(self.device)

                    context_size = tokenized["input_ids"].shape[1]
                    if context_size > self.max_length:
                        max_new_tokens = 1
                    elif max_new_tokens is None:
                        max_new_tokens = self.max_length - context_size
                    else:
                        max_new_tokens = min(self.max_length - context_size, max_new_tokens)
                        if max_new_tokens < 1:
                            max_new_tokens = 1
                    if self.generation_cap is not None:  # FIX 2: cap generation length
                        max_new_tokens = min(max_new_tokens, self.generation_cap)

                    prepared_batch = _tm.Batch(
                        input_ids=tokenized["input_ids"],
                        input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                        input_mask=tokenized["attention_mask"],
                        truncated=[max(len(c) - tokenized["input_ids"].shape[1], 0) for c in context],
                        padded=[sum(mask == 0) for mask in tokenized["attention_mask"]],
                    )
                    cur_responses = self._generate(
                        batch=prepared_batch, max_new_tokens=max_new_tokens, stop_tokens=stop_tokens,
                        returns_logits=returns_logits, num_samples=num_samples, do_sample=do_sample,
                    )
                    results.extend(cur_responses)

            return dataset.get_original_order(results)

    return OzakiTransformersModel


def main():
    args = parse_args()
    if not args.overwrite and os.path.exists(args.output_path):
        print(f"Results found at {args.output_path}. Skip. (use --overwrite to redo)")
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from accelerate import Accelerator, InitProcessGroupKwargs
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval_custom.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    tasks, custom_tasks = DATASET_TASKS[args.dataset]

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    env_config = EnvConfig(token=os.getenv("HF_TOKEN"), cache_dir=os.getenv("HF_HOME", "/data/hf_cache/hub"))

    # top_k=0 disables top-k (matches the vLLM path, which used top_k=None for non-QwQ).
    generation_parameters = GenerationParameters(
        temperature=args.temperature, top_p=args.top_p, top_k=0,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    model_config = TransformersModelConfig(
        pretrained=args.model,
        dtype="bfloat16",
        accelerator=accelerator,
        model_parallel=False,
        max_length=args.max_model_length,
        batch_size=args.batch_size,
        use_chat_template=True,
        generation_parameters=generation_parameters,
    )

    OzakiModel = make_ozaki_model_class(args.nmp, args.rslt_type, args.chunk_size,
                                        sanitize_logits=not args.no_sanitize_logits,
                                        linear_only=args.linear_only, attn_impl=args.attn_impl,
                                        gemm_bits=args.gemm_bits,
                                        byte_split_style=args.byte_split_style)
    model = OzakiModel(env_config, model_config)
    model.generation_cap = args.max_new_tokens  # cap generated tokens (emulation is slow)

    evaluation_tracker = EvaluationTracker(output_dir=args.output_dir, save_details=False)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        env_config=env_config,
        custom_tasks_directory=custom_tasks,
        override_batch_size=args.batch_size,
        num_fewshot_seeds=1,
        max_samples=args.max_samples,
        use_chat_template=True,
    )
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
        metric_options={},
    )

    # Data-parallel sharding: keep a disjoint stride-subset of requests for this shard
    # (union over shards = all requests; each processed exactly once). docs dict is left
    # whole so metric lookups still resolve.
    if args.num_shards > 1:
        kept = 0
        for rt in list(pipeline.requests.keys()):
            pipeline.requests[rt] = [r for i, r in enumerate(pipeline.requests[rt])
                                     if i % args.num_shards == args.shard_id]
            kept += len(pipeline.requests[rt])
        print(f"[shard {args.shard_id}/{args.num_shards}] processing {kept} requests")

    pipeline.evaluate()
    pipeline.show_results()

    # Save per-sample results in the same JSON format as inference_vllm.py.
    details = evaluation_tracker.details
    task_name = list(details.keys())[0]
    eval_results = []
    for detail in details[task_name]:
        eval_results.append({
            "full_prompt": detail["full_prompt"],
            "generated_text": detail["predictions"][0],
            "gold": detail["gold"],
            "metrics": detail["metrics"],
        })
    with open(args.output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved {len(eval_results)} results to {args.output_path}")


if __name__ == "__main__":
    main()
