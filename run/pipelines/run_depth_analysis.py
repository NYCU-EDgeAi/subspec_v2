"""Depth analysis pipeline for collecting acceptance length statistics."""

import os
import json
import time
import torch
import gc
import logging
import numpy as np
from tqdm import tqdm, trange
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import LogitsProcessorList

from .benchmarks.registry import load_dataset, validate_benchmarks, extract_prompt
from .utils.benchmark_utils import (
    cleanup_gpu,
    parse_benchmark_list,
    reset_seeds,
    setup_benchmark_dir,
)
from .utils.eval_utils import reset_kv


def run_depth_analysis_on_dataset(
    generator,
    tokenizer,
    past_kv,
    dataset,
    max_depth,
    args,
    log_dir,
    save_samples: bool = True,
):
    """
    Run depth analysis on a dataset, collecting acceptance length statistics.
    
    Args:
        generator: The speculative decoding generator
        tokenizer: Tokenizer for the model
        past_kv: KV cache for the target model
        dataset: List of prompts/items to process
        max_depth: Maximum draft tree depth
        args: Configuration arguments
        log_dir: Directory to save results
        
    Returns:
        dict: Summary metrics
    """
    device = args.device
    all_acceptance_lengths = []
    prompt_sample_counts = []  # Track samples per prompt (compact)
    
    # Detailed sample logs for comparison with benchmark
    sample_logs = []
    
    pbar = tqdm(dataset, desc="Depth Analysis")
    for prompt_idx, item in enumerate(pbar):
        prompt = extract_prompt(item)
        prompt_samples = 0  # Count samples for this prompt
        prompt_acceptance_lengths = []  # Track acceptance per iteration for this prompt
        
        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        tokenizer.use_default_system_prompt = True
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        _, org_input_len = input_ids.shape
        
        logging.debug(f"[Prompt {prompt_idx}] Input length: {org_input_len} tokens")
        logging.debug(f"[Prompt {prompt_idx}] Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        
        # Skip if too long
        if input_ids.shape[1] > args.max_length:
            logging.info(f"Skipping prompt {prompt_idx} due to length {input_ids.shape[1]} > {args.max_length}")
            prompt_sample_counts.append(0)
            if save_samples:
                sample_logs.append({
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "input_len": int(org_input_len),
                    "skipped": True,
                    "skip_reason": "too_long",
                })
            continue
        
        # Reset cache
        past_kv.reset()
        generator.draft_model.set_past_key_values(past_kv)
        
        # Init tree mask
        max_verify = generator.draft_params.max_verify_tokens
        max_cache_len = getattr(past_kv, "max_cache_len", None)
        if max_cache_len is None and hasattr(past_kv, "cache"):
            max_cache_len = getattr(past_kv.cache, "max_cache_len", 4096)
        max_cache_len = max_cache_len or 4096
        generator._init_tree_mask(max_verify, max_cache_len, device=device)
        
        generated_token_ids = []
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            # Prefill
            outputs = generator._chunked_prefill_forward(
                input_ids, past_kv,
                prefill_chunk_size=generator.prefill_chunk_size or 256,
                use_position_ids=True,
            )
            
            # First token
            logits_processor = LogitsProcessorList()
            first_token = generator._sample_token(outputs.logits, logits_processor, do_sample=False)
            input_ids = torch.cat([input_ids, first_token], dim=-1)
            generated_token_ids.append(first_token.item())
            cache_position = torch.arange(org_input_len, org_input_len + max_verify, dtype=torch.long, device=device)
            del outputs
            
            # Main loop - run until EOS or max_length reached
            while input_ids.shape[1] < args.max_length:
                last_token = input_ids[:, -1:].clone(memory_format=torch.contiguous_format)
                prev_kv_len = past_kv.get_seq_length()
                
                # Draft
                tree = generator._speculate(last_token)
                
                # Target forward on tree
                position_offset = input_ids.shape[1] - 1
                outputs = generator._tree_decoding(tree, past_kv, position_offset=position_offset,
                                                   cache_position=cache_position, device=device)
                
                # Verify
                sampled_tokens, hidden_indices, (_, accept_len) = generator._verify(
                    tree, root_ind=0, logits=outputs.logits, logits_processor=logits_processor, do_sample=False
                )
                all_acceptance_lengths.append(accept_len)
                prompt_acceptance_lengths.append(accept_len)
                prompt_samples += 1
                
                target_token = sampled_tokens[0:1, 0:1].to(device)
                generated_token_ids.append(target_token.item())
                
                # Update progress
                gen_len = input_ids.shape[1] - org_input_len
                pbar.set_postfix({"gen": gen_len, "acc": accept_len, "samples": len(all_acceptance_lengths)})
                
                del outputs
                
                # Reorder KV cache and add single token
                # past_kv.reorder_cache_with_offset(hidden_indices[:1], offset=prev_kv_len,                               new_chunk_len=generator.draft_params.max_verify_tokens, dim=2)
                
                # crop to seq_len + 1
                past_kv.seq_len += 1
                past_kv.crop(past_kv.seq_len, past_kv.seq_len+max_depth, dim=2)
                
                # Update cache_position for next iteration
                cache_position = cache_position + 1
                
                input_ids = torch.cat([input_ids, target_token], dim=-1)
                
                # Check for EOS
                if target_token.item() == tokenizer.eos_token_id:
                    break
                if tokenizer.eos_token_id in sampled_tokens:
                    # append al sampled_tokens to generated_token_ids
                    generated_token_ids.extend(sampled_tokens[0, 1:].tolist())
                    break
        
        prompt_sample_counts.append(prompt_samples)
        
        # Decode generated text and log
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        logging.debug(f"[Prompt {prompt_idx}] Generated {len(generated_token_ids)} tokens: "
                      f"'{generated_text[:300]}{'...' if len(generated_text) > 300 else ''}'")
        
        # Save detailed sample info
        if save_samples:
            sample_logs.append({
                "prompt_idx": prompt_idx,
                "query": prompt,
                "input_len": int(org_input_len),
                "skipped": False,
                "n_tokens": len(generated_token_ids),
                "n_iterations": prompt_samples,
                "acceptance_lengths": prompt_acceptance_lengths,
                "mean_acceptance": float(np.mean(prompt_acceptance_lengths)) if prompt_acceptance_lengths else 0.0,
                "response": generated_text,
            })
        
        # Log progress
        logging.debug(f"Prompt {prompt_idx + 1}: {prompt_samples} samples, "
                      f"{len(generated_token_ids)} tokens generated")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Convert to numpy arrays
    acc_arr = np.array(all_acceptance_lengths, dtype=np.int32)
    prompt_counts = np.array(prompt_sample_counts, dtype=np.int32)
    
    # Compute summary statistics
    if len(acc_arr) > 0:
        mean_acc = float(acc_arr.mean())
        std_acc = float(acc_arr.std())
        
        # Compute tau(D) and beta(D) for common depths
        depths = [8, 16, 32, 64]
        tau_values = {}
        beta_values = {}
        for d in depths:
            if d <= max_depth:
                tau_values[f"tau_{d}"] = float(np.mean(np.minimum(acc_arr, d)))
                beta_values[f"beta_{d}"] = float(np.mean(acc_arr >= d))
    else:
        mean_acc = 0.0
        std_acc = 0.0
        tau_values = {}
        beta_values = {}
    
    # Save numpy arrays
    np.save(os.path.join(log_dir, "acc_arr.npy"), acc_arr)
    np.save(os.path.join(log_dir, "prompt_sample_counts.npy"), prompt_counts)
    
    # Save logs in run_benchmark style: one JSON object per line, compact arrays, minimal fields
    if save_samples and sample_logs:
        log_file = os.path.join(log_dir, "0.jsonl")
        for i, sample in enumerate(sample_logs):
            exp_log = {
                "query": sample["query"],
                "response": sample["response"],
                "n_tokens": sample["n_tokens"],
                "n_iterations": sample["n_iterations"],
                "acceptance_lengths": sample["acceptance_lengths"],
                "mean_acceptance": sample["mean_acceptance"],
            }
            with open(log_file, 'a+') as f:
                # Dump with compact separators for arrays
                json.dump(exp_log, f, indent=4)
                f.write("\n")
        # Add overall summary as last line
        overall_log = {
            "overall": True,
            "n_prompts": len(sample_logs),
            "n_samples": int(np.sum([s["n_iterations"] for s in sample_logs])),
            "mean_acceptance": float(np.mean([s["mean_acceptance"] for s in sample_logs])) if sample_logs else 0.0,
            "mean_n_tokens": float(np.mean([s["n_tokens"] for s in sample_logs])) if sample_logs else 0.0,
        }
        with open(log_file, 'a+') as f:
            json.dump(overall_log, f, indent=4)
            f.write("\n")
        logging.info(f"Saved {len(sample_logs)} sample logs to {log_file}")
    
    # Build results summary
    results = {
        "n_samples": len(acc_arr),
        "n_prompts": len(prompt_counts),
        "n_prompts_with_data": int(np.sum(prompt_counts > 0)),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "min_acc": int(acc_arr.min()) if len(acc_arr) > 0 else 0,
        "max_acc": int(acc_arr.max()) if len(acc_arr) > 0 else 0,
        "max_depth": max_depth,
        **tau_values,
        **beta_values,
        "samples_per_prompt_mean": float(prompt_counts.mean()) if len(prompt_counts) > 0 else 0,
        "samples_per_prompt_std": float(prompt_counts.std()) if len(prompt_counts) > 0 else 0,
    }
    
    # Save results JSON
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Depth Analysis Results:")
    print(f"  Total samples: {results['n_samples']}")
    print(f"  Prompts processed: {results['n_prompts_with_data']}/{results['n_prompts']}")
    print(f"  Mean acceptance length: {mean_acc:.2f} ± {std_acc:.2f}")
    if "tau_32" in results:
        print(f"  τ(32): {results['tau_32']:.2f}")
        print(f"  β(32): {results['beta_32']:.2%}")
    print(f"  Saved to: {log_dir}")
    print(f"{'='*60}\n")
    
    return results


def main(builder, benchmarks=None, max_samples=None):
    """Run depth analysis on specified benchmarks."""
    reset_seeds(0)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    
    builder.generator_profiling = False
    builder.profiling_verbose = False
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    max_depth = generator.draft_params.max_depth
    
    # Validate benchmarks
    bench_list = parse_benchmark_list(benchmarks)
    validate_benchmarks(bench_list)
    
    print(f"Benchmarks to analyze: {bench_list}")
    print(f"Max samples per benchmark: {max_samples or 'all'}")
    print(f"Max depth: {max_depth}")
    
    # Warmup
    if args.warmup_iter > 0:
        print(f"\nWarming up ({args.warmup_iter} iterations)...")
        is_profiling = generator.profiling
        generator.profiling = False
        
        for i in trange(args.warmup_iter, desc='Warming up'):
            input_message = "Write an essay about large language models."
            messages = [{"role": "user", "content": input_message}]
            tokenizer.use_default_system_prompt = True
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=True
            ).cuda(device=args.device)
            
            cleanup_gpu()
            
            with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
                generator.generate(
                    input_ids,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    past_key_values=past_kv,
                    draft_past_key_values=draft_past_kv
                )
            
            reset_kv(past_kv, draft_past_kv)
        
        generator.profiling = is_profiling
    
    # Run analysis on each benchmark
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_depth_analysis")
    all_results = {}
    for bench_name in tqdm(bench_list, desc="Benchmarks"):
        reset_seeds(0)
        log_dir = setup_benchmark_dir(log_dir_base, bench_name, getattr(args, "settings_snapshot", None))
        print(f"\n{'='*60}")
        print(f"Analyzing benchmark: {bench_name}")
        print(f"Output directory: {log_dir}")
        
        dataset = load_dataset(bench_name, max_samples=max_samples, seed=0, shuffle=True)
        print(f"Samples: {len(dataset)}")
        
        cleanup_gpu()
        reset_kv(past_kv, draft_past_kv)
        
        results = run_depth_analysis_on_dataset(
            generator=generator,
            tokenizer=tokenizer,
            past_kv=past_kv,
            dataset=dataset,
            max_depth=max_depth,
            args=args,
            log_dir=log_dir,
        )
        
        all_results[bench_name] = results
        cleanup_gpu()
    
    return all_results
