import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import trange
import logging
import os
import nvtx
import random
import time
import json
from specdecodes.models.utils.wandb_logger import wandb_logger
from run.pipelines.utils.eval_utils import reset_kv, maybe_init_cuda_graph_runner
from run.core.config_utils import write_settings_yaml


def _build_compact_results(exp_log: dict) -> dict:
    return {
        "tput": float(exp_log.get("tput", 0.0) or 0.0),
        "avg_sampled": float(exp_log.get("avg_sampled", 0.0) or 0.0),
        "verify_nonterminal_rounds": int(exp_log.get("verify_nonterminal_rounds", 0) or 0),
        "mean_verify_accept_len_nonterminal": float(
            exp_log.get("mean_verify_accept_len_nonterminal", 0.0) or 0.0
        ),
        "std_verify_accept_len_nonterminal": float(
            exp_log.get("std_verify_accept_len_nonterminal", 0.0) or 0.0
        ),
        "avg_draft_time": float(exp_log.get("avg_draft_time", 0.0) or 0.0),
        "avg_target_time": float(exp_log.get("avg_target_time", 0.0) or 0.0),
        "avg_verify_time": float(exp_log.get("avg_verify_time", 0.0) or 0.0),
        "post_verify_count": int(exp_log.get("post_verify_count", 0) or 0),
        "speculate_count": int(exp_log.get("speculate_count", 0) or 0),
        "post_verify_rate": float(exp_log.get("post_verify_rate", 0.0) or 0.0),
        "is_prev_accepted_count": int(exp_log.get("is_prev_accepted_count", 0) or 0),
        "is_prev_accepted_steps": int(exp_log.get("is_prev_accepted_steps", 0) or 0),
        "is_prev_accepted_rate": float(exp_log.get("is_prev_accepted_rate", 0.0) or 0.0),
        "n_prompt_tokens": int(exp_log.get("n_prompt_tokens", 0) or 0),
        "n_output_tokens": int(exp_log.get("n_output_tokens", 0) or 0),
        "elapsed_time": float(exp_log.get("elapsed_time", 0.0) or 0.0),
        "peak_memory_gib": float(exp_log.get("peak_memory", 0.0) or 0.0),
    }


def _print_compact_results(results: dict) -> None:
    print("Final run_test Results:")
    print(f"\tThroughput       : {results['tput']:.3f} tokens/sec")
    print(f"\tToken Acceptance : {results['avg_sampled']:.3f} (avg_sampled)")
    rounds = int(results["verify_nonterminal_rounds"])
    if rounds > 0:
        print(
            f"\tVerify Accept Len (nonterminal): {results['mean_verify_accept_len_nonterminal']:.3f} "
            f"± {results['std_verify_accept_len_nonterminal']:.3f} "
            f"tokens/verify ({rounds} rounds)"
        )
    print(f"\tAvg Draft Time   : {results['avg_draft_time']:.3f} sec")
    print(f"\tAvg Target Time  : {results['avg_target_time']:.3f} sec")
    print(f"\tAvg Verify Time  : {results['avg_verify_time']:.3f} sec")
    print(f"\tPost-Verify Rate : {results['post_verify_rate']:.3f}")
    print(
        f"\tPrevAccepted     : {int(results['is_prev_accepted_count'])}/"
        f"{int(results['is_prev_accepted_steps'])} ({results['is_prev_accepted_rate']:.3f})"
    )
    print(f"\tOutput Tokens    : {int(results['n_output_tokens'])}")
    print(f"\tPeak Memory      : {results['peak_memory_gib']:.3f} GiB")


def main(builder):
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    # Suppress verbose per-depth profiling tables in run_test; emit compact summary instead.
    generator.profiling_verbose = False

    # deterministic
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # warm up
    if args.warmup_iter > 0:
        print("Warming up... It will take some time for the first few iterations to run.")
        with nvtx.annotate("Warming up"):
            is_profiling = generator.profiling
            generator.profiling = False
            for i in trange(args.warmup_iter, desc='Warming up'):
                input_message = "Write an essay about large language models."
                messages = [{"role": "user", "content": input_message}]
                tokenizer.use_default_system_prompt = True
                with nvtx.annotate("Warm up"):
                    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(args.device)
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_kv, draft_past_key_values=draft_past_kv)
                
                reset_kv(past_kv, draft_past_kv)
            generator.profiling = is_profiling

    # Optional CUDA-graph capture for FlashInfer, after warmup (stabilizes kernels/allocations).
    maybe_init_cuda_graph_runner(generator, past_kv, draft_past_kv, args.device, args.warmup_iter)
        
    # input message
    input_message = "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?"
    # input_message = "Describe what is large language models to me."
    messages = [{"role": "user", "content": input_message}]
    tokenizer.use_default_system_prompt = True
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(args.device)
    prompt = tokenizer.decode(input_ids[0])
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
                  
    # generate response
    print("Generating response...")
    torch.cuda.cudart().cudaProfilerStart() # start profiling from here
    start_event.record()
    with nvtx.annotate("Generate"):
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_kv, draft_past_key_values=draft_past_kv)
    end_event.record()
    
    # Ensure all CUDA kernels are done.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    total_time_s = start_event.elapsed_time(end_event) / 1000.0
    output = generator.tokenizer.decode(output_ids[0][input_ids.shape[1]:])

    # Persist a single-run log (mirrors benchmark JSONL style).
    log_dir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_test")
    os.makedirs(log_dir, exist_ok=True)
    write_settings_yaml(log_dir, getattr(args, "settings_snapshot", None))
    log_file = os.path.join(log_dir, "0.jsonl")
    exp_log = {
        **wandb_logger.log_data,
        "input_message": input_message,
        "prompt": prompt,
        "response": output,
        "elapsed_time": float(total_time_s),
        "n_prompt_tokens": int(input_ids.shape[1]),
        "n_output_tokens": int(output_ids.shape[1] - input_ids.shape[1]),
        "peak_memory": float(torch.cuda.max_memory_reserved(args.device) / (1024**3)),
    }
    with open(log_file, "a+", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=4)
        f.write("\n")

    compact_results = _build_compact_results(exp_log)
    results_file = os.path.join(log_dir, "results.jsonl")
    with open(results_file, "a+", encoding="utf-8") as f:
        json.dump({"run_test": compact_results}, f, indent=4)
        f.write("\n")
    print(f"Log directory: {log_dir}")
    _print_compact_results(compact_results)

    if args.print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if args.print_time:
        print("Time:", total_time_s)
