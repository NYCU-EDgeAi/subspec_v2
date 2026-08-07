"""Unified experiment sweep — one base config x a set of `--set`-style override points.

A sweep is declared once in a small YAML (see configs/sweeps/*.yaml):

    base: configs/methods/subspec_sd.yaml   # optional; else the top-level --config
    run: run-benchmark                       # inner action per point
    set:   {max_length: 512}                 # fixed overrides on every point
    axes:  {draft_params.max_depth: [8, 16]} # cartesian product
    include:                                 # correlated bundles (travel together)
      - {llm_path: Qwen/Qwen2.5-7B-Instruct, vram_limit_gb: 8}

Points = `include` x cartesian(`axes`); each point is `base` deep-merged with its overrides,
which is exactly `base config + a set of --set tuples` (this reuses run.main's real
`--set`/merge machinery). Points that touch a model-construction field trigger a full
model rebuild; the rest reuse the resident target/draft weights and only rebuild the
generator pipeline (à la run_grid_search). Per point writes settings.yaml + results.jsonl;
the sweep writes an index.jsonl for discoverability.
"""
import gc
import itertools
import json
import logging
import os
import random
import time

import yaml
import torch

from run.main import _load_yaml_config, _deep_merge_dict, _apply_yaml_overrides
from run.core.registry import ModelRegistry
from run.core.presets import register_presets
from run.core.configuration import AppConfig
from run.core.config_utils import instantiate_recipe, write_settings_yaml
from run.core.builder import GeneratorPipelineBuilder
from run.pipelines.benchmarks.utils.eval import run_mtbench_eval
from run.pipelines.benchmarks.mtbench import load_mtbench_dataset

# Overriding any of these forces a full model rebuild; everything else is a cheap
# generator-pipeline rebuild that reuses the resident target/draft weights.
MODEL_FIELDS = {
    "method", "backend", "llm_path", "draft_model_path", "recipe",
    "dtype", "device", "vram_limit_gb", "cpu_offload_gb",
}


def _nest(dotted: str, value):
    node: dict = {}
    cur = node
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return node


def _to_nested(pairs: dict) -> dict:
    out: dict = {}
    for dotted, value in pairs.items():
        out = _deep_merge_dict(out, _nest(dotted, value))
    return out


def _expand_points(spec: dict):
    """Yield (label, point_pairs) for every sweep point.

    point_pairs is a flat {dotted_key: value} of the fixed `set` + one `include` bundle
    + one `axes` combination."""
    fixed = dict(spec.get("set", {}) or {})
    axes = dict(spec.get("axes", {}) or {})
    include = list(spec.get("include", []) or [{}])
    axis_keys = list(axes)
    axis_combos = list(itertools.product(*[axes[k] for k in axis_keys])) if axis_keys else [()]
    for bundle in include:
        for combo in axis_combos:
            varying = {**dict(bundle), **dict(zip(axis_keys, combo))}
            pairs = {**fixed, **varying}
            label = "__".join(f"{k}={v}" for k, v in varying.items()) or "point"
            # keep the label filesystem-safe
            label = label.replace("/", "-").replace(" ", "")
            yield label, pairs


def _model_signature(pairs: dict, base_yaml: dict) -> str:
    """Stable signature over the model-construction fields for a point."""
    merged = _deep_merge_dict(base_yaml, _to_nested(pairs))
    sig = {k: merged.get(k) for k in sorted(MODEL_FIELDS)}
    return json.dumps(sig, sort_keys=True, default=str)


def _build_point_config(base_yaml: dict, base_method: str, base_path: str, pairs: dict) -> AppConfig:
    """base config + point overrides -> a resolved AppConfig (mirrors run.main's build)."""
    point_yaml = _deep_merge_dict(base_yaml, _to_nested(pairs))
    method = point_yaml.get("method") or base_method
    default_config = ModelRegistry.get(method).default_config.copy()
    default_config = _apply_yaml_overrides(
        default_config, {k: v for k, v in point_yaml.items() if k != "method"}
    )
    cfg = AppConfig()
    cfg.method = method
    cfg.update(default_config)  # rejects unknown keys (typo guard)
    cfg.recipe = instantiate_recipe(getattr(cfg, "recipe", None))
    cfg.config_path = base_path
    return cfg


def _reset_compile_caches():
    for reset in (
        lambda: torch.compiler.reset(),
        lambda: __import__("torch._dynamo", fromlist=["reset"]).reset(),
    ):
        try:
            reset()
        except Exception:
            pass


def main(builder, spec_path: str, max_samples: int = None):
    register_presets()
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    spec = yaml.safe_load(open(spec_path))
    base_path = spec.get("base") or getattr(builder.config, "config_path", None)
    if base_path is None:
        raise ValueError("Sweep needs a base config: set `base:` in the sweep YAML or pass --config.")
    base_yaml = _load_yaml_config(base_path)
    base_method = base_yaml.get("method") or builder.config.method
    run = spec.get("run", "run-benchmark")
    if run != "run-benchmark":
        raise NotImplementedError(
            f"run-sweep currently supports run: run-benchmark (got {run!r}); run-test is a follow-up."
        )

    points = list(_expand_points(spec))
    # Order points so those sharing a model config are adjacent -> fewer full reloads.
    points.sort(key=lambda lp: _model_signature(lp[1], base_yaml))

    sweep_dir = os.path.join(
        builder.config.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_sweep"
    )
    os.makedirs(sweep_dir, exist_ok=True)
    index_path = os.path.join(sweep_dir, "index.jsonl")
    logging.info("Sweep: %d point(s) from %s (base %s) -> %s", len(points), spec_path, base_path, sweep_dir)

    dataset = load_mtbench_dataset()
    n_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    torch.manual_seed(0)
    random.seed(0)
    random.shuffle(dataset)
    dataset = dataset[:n_samples]

    model = draft_model = tokenizer = None
    base_target_forward = base_draft_forward = None
    last_sig = None

    for idx, (label, pairs) in enumerate(points):
        cfg = _build_point_config(base_yaml, base_method, base_path, pairs)
        pbuilder = GeneratorPipelineBuilder(cfg)
        sig = _model_signature(pairs, base_yaml)

        torch.manual_seed(0)
        random.seed(0)
        _reset_compile_caches()
        torch.cuda.empty_cache()
        gc.collect()

        rebuilt_model = sig != last_sig or model is None
        if rebuilt_model:
            model, draft_model, tokenizer = pbuilder.build_models_and_tokenizer()
            base_target_forward = model.forward
            base_draft_forward = draft_model.forward if draft_model is not None else None
            last_sig = sig
        else:
            # Reuse resident weights; restore pristine forwards so torch.compile is clean.
            model.forward = base_target_forward
            if draft_model is not None and base_draft_forward is not None:
                draft_model.forward = base_draft_forward

        logging.info("[%d/%d] %s  (%s)", idx + 1, len(points), label,
                     "full rebuild" if rebuilt_model else "reuse model")

        log_dir = os.path.join(sweep_dir, f"{idx:03d}_{label}"[:200])
        os.makedirs(log_dir, exist_ok=True)
        snapshot = pbuilder.settings_snapshot or {}
        write_settings_yaml(log_dir, {**snapshot, "sweep_point": pairs})

        generator, tokenizer, past_kv, draft_past_kv = pbuilder.build_generator_pipeline(
            model, draft_model, tokenizer
        )

        status, metrics = "ok", {}
        try:
            metrics = run_mtbench_eval(generator, tokenizer, past_kv, draft_past_kv,
                                       pbuilder.args, dataset, log_dir) or {}
        except Exception as e:  # keep the sweep going; record the failure
            status = f"error: {e}"
            logging.warning("point %s failed: %s", label, e)

        with open(os.path.join(log_dir, "results.jsonl"), "w") as f:
            json.dump({"point": pairs, "status": status, "metrics": metrics}, f, indent=2)
        with open(index_path, "a") as f:
            f.write(json.dumps({
                "idx": idx,
                "label": label,
                "point": pairs,
                "rebuilt_model": rebuilt_model,
                "status": status,
                "tput_mean": metrics.get("tput_mean"),
                "tacc_mean": metrics.get("tacc_mean"),
                "dir": os.path.relpath(log_dir, sweep_dir),
            }) + "\n")

        _reset_compile_caches()
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Sweep done. Index: %s", index_path)
    print(f"\nSweep complete: {len(points)} point(s). Index -> {index_path}")
