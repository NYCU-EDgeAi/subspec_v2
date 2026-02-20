# Benchmarking Protocol (fp16 vs int4 vs Lossy-SD)

This repo now uses a lane-based benchmarking flow with one evaluator registry (`run/pipelines/benchmarks/task_registry.py`) and two execution pipelines:

- `run-benchmark-acc`: single-model accuracy/perf evaluation.
- `run-benchmark-compare`: paired comparison (flips + KL/task-native metrics).

## Lanes

### Distribution lane (`distribution`)

- LL-based scoring (canonical for multiple-choice tasks).
- For MC tasks, reports `accuracy`, `acc_norm`.
- In compare mode, reports paper-aligned flip metrics:
  - `flips_rate`
  - `flips_c2i_count` / `flips_i2c_count`
  - `allflips_rate`
  - plus `kl_choice`, optional `kl_token`.
- Primary use: fp16 vs int4 or any weight-change comparison.
- MC tasks (`hellaswag,piqa,arc-c,winogrande`) are LL-only in this repo.

### Behavior lane (`behavior`)

- Generation-based scoring (exercises decoding behavior, including SD paths).
- Used for SD behavior-change analysis.
- In compare mode, reports task-native accuracy + flips for:
  - `gsm8k`
  - `human-eval`
  - `human-eval` also reports `output_change_rate`.

## Recommended Benchmark Sets

- **Distribution lane (MC):** `hellaswag,piqa,arc-c,winogrande`
- **Behavior lane (decode-dominant):** `gsm8k,human-eval`

## Commands

### Single-model accuracy (`run-benchmark-acc`)

```bash
python -m run.main \
  --config configs/methods/subspec_sd_no_offload.yaml \
  run-benchmark-acc \
  --benchmarks hellaswag,piqa,arc-c,winogrande \
  --lane distribution \
  --max-samples 200
```

```bash
python -m run.main \
  --config configs/methods/subspec_sd_no_offload.yaml \
  run-benchmark-acc \
  --benchmarks gsm8k,human-eval \
  --lane behavior \
  --max-samples 200
```

### Paired compare (`run-benchmark-compare`)

fp16 vs int4:

```bash
python -m run.main \
  --config configs/methods/subspec_sd_no_offload.yaml \
  run-benchmark-compare \
  --compare-config configs/methods/vanilla_int4.yaml \
  --compare-name int4 \
  --benchmarks hellaswag,piqa,arc-c,winogrande \
  --lane distribution \
  --max-samples 200
```

fp16 vs Lossy-SD:

```bash
python -m run.main \
  --config configs/methods/subspec_sd_no_offload.yaml \
  run-benchmark-compare \
  --compare-config configs/methods/subspec_sd_lossy_no_offload.yaml \
  --compare-name lossy_sd \
  --benchmarks hellaswag,piqa,arc-c,winogrande \
  --lane distribution \
  --max-samples 200
```

fp16 vs Lossy-SD (decode-dominant acc+flips):

```bash
python -m run.main \
  --config configs/methods/subspec_sd_no_offload.yaml \
  run-benchmark-compare \
  --compare-config configs/methods/subspec_sd_lossy_no_offload.yaml \
  --compare-name lossy_sd \
  --benchmarks gsm8k,human-eval \
  --lane behavior \
  --max-samples 200
```

Reuse an existing baseline (skip re-running base model):

```bash
python -m run.main \
  --config configs/methods/subspec_sd_no_offload.yaml \
  run-benchmark-compare \
  --compare-config configs/methods/vanilla_int4.yaml \
  --compare-name int4 \
  --benchmarks hellaswag,piqa,arc-c,winogrande \
  --lane distribution \
  --reuse-baseline-dir experiments/<timestamp>/run_benchmark_compare \
  --max-samples 200
```

Notes:
- Reuse mode writes to a new timestamped output directory and copies baseline JSONL files there.
- Token-level KL is disabled when reusing baselines (requires in-memory baseline logits).
- `run-benchmark-compare` supports only `hellaswag,piqa,arc-c,winogrande,gsm8k,human-eval`.
- MC tasks are LL-only and only support `--lane distribution`.
- `gsm8k` and `human-eval` task-native compare is behavior-lane only.

## Three-way workflow (fp16, int4, Lossy-SD)

Use fp16 as the common baseline:

1. Run fp16 vs int4 (`distribution` lane, MC tasks).
2. Reuse the same fp16 baseline dir and run fp16 vs Lossy-SD (`behavior` lane, SD-sensitive tasks).
3. Compare `flips_c2i_count/flips_i2c_count/allflips_rate` across both compare outputs.

## Output Files

Per benchmark directory:

- `<bench>_base.jsonl`: baseline per-sample outputs/scores.
- `<bench>_compare.jsonl`: paired per-sample comparison records.
- `results.jsonl`: aggregate metrics for each compare run.
- `settings.yaml`: run config snapshot.

## Source of Truth

- Loader registry: `run/pipelines/benchmarks/registry.py`
- Task-to-lane evaluator mapping: `run/pipelines/benchmarks/task_registry.py`
- Evaluators: `run/pipelines/benchmarks/utils/eval_acc.py`
- Acc pipeline: `run/pipelines/run_benchmark_acc.py`
- Compare pipeline: `run/pipelines/run_benchmark_compare.py`

Each benchmark loader includes top-level comments with the canonical implementation reference.
