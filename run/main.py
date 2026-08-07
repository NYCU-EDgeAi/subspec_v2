from __future__ import annotations

import sys
import argparse
import os
import shutil
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .core.configuration import AppConfig


BENCHMARK_COMMANDS = {"run-benchmark", "run-benchmark-agent", "run-benchmark-compare"}


def _configure_allocator_env(default: str = "expandable_segments:True") -> None:
    """Configure PyTorch allocator env vars.

    Some PyTorch builds still apply CUDA allocator settings more reliably via
    PYTORCH_CUDA_ALLOC_CONF, while newer versions encourage PYTORCH_ALLOC_CONF.
    We support both by mirroring values and providing stable defaults.
    """

    if "PYTORCH_ALLOC_CONF" in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
        return

    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        return

    os.environ.setdefault("PYTORCH_ALLOC_CONF", default)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", default)


def _maybe_patch_auto_gptq() -> None:
    """Monkey patch for auto_gptq compatibility with optimum (best-effort)."""

    try:
        import auto_gptq  # type: ignore[import-not-found]

        if not hasattr(auto_gptq, "QuantizeConfig") and hasattr(auto_gptq, "BaseQuantizeConfig"):
            auto_gptq.QuantizeConfig = auto_gptq.BaseQuantizeConfig
    except ImportError:
        pass


def _configure_runtime_environment() -> None:
    # Reduce run-to-run drift from cuBLAS matmul reductions.
    # Important: set before the first CUDA context initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    # Keep allocator behavior stable by default (can be overridden via env).
    _configure_allocator_env(default="expandable_segments:True")


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not override:
        return dict(base)
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _parse_set_overrides(pairs) -> Dict[str, Any]:
    """Parse repeated ``--set key.path=value`` flags into one nested dict.

    The value is parsed as a YAML scalar so `16`->int, `1e-3`->float, `true`->bool,
    `null`->None, `[a,b]`->list; anything else stays a string. Dotted keys nest, e.g.
    ``draft_params.max_depth=8`` -> ``{"draft_params": {"max_depth": 8}}``. The result is
    deep-merged over the YAML config, so it flows through the normal draft_params /
    generator_kwargs handling. (Merging into a `recipe:` the method preset supplies as an
    object won't work; specify `recipe:` as a YAML block to override its kwargs.)
    """
    import yaml

    result: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY.PATH=VALUE, got {item!r}")
        key_path, _, raw = item.partition("=")
        key_path = key_path.strip()
        if not key_path:
            raise ValueError(f"--set has an empty key in {item!r}")
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        node = result
        parts = key_path.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return result


def _draft_params_to_dict(dp) -> Dict[str, Any]:
    if dp is None:
        return {}
    if is_dataclass(dp):
        return dict(asdict(dp))
    if hasattr(dp, "__dict__"):
        return dict(dp.__dict__)
    return {}


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _serialize_recipe(recipe: Any) -> Any:
    if recipe is None:
        return None
    if isinstance(recipe, (str, dict)):
        return _to_serializable(recipe)

    class_path = f"{recipe.__class__.__module__}:{recipe.__class__.__name__}"
    payload: Dict[str, Any] = {"class_path": class_path}
    if hasattr(recipe, "__dict__"):
        payload["kwargs"] = _to_serializable(recipe.__dict__)
    return payload


def _build_settings_snapshot(
    *,
    config: "AppConfig",
    config_path: str | None,
    subcommand_argv: list[str],
) -> Dict[str, Any]:
    generator_kwargs = dict(getattr(config, "generator_kwargs", {}) or {})
    draft_params = _draft_params_to_dict(getattr(config, "draft_params", None))

    snapshot: Dict[str, Any] = {
        "config_path": config_path,
        "subcommand": subcommand_argv[0] if subcommand_argv else None,
        "subcommand_args": subcommand_argv[1:] if len(subcommand_argv) > 1 else [],
        "method": getattr(config, "method", None),
        "llm_path": getattr(config, "llm_path", None),
        "draft_model_path": getattr(config, "draft_model_path", None),
        "device": _to_serializable(getattr(config, "device", None)),
        "dtype": _to_serializable(getattr(config, "dtype", None)),
        "seed": getattr(config, "seed", None),
        "max_length": getattr(config, "max_length", None),
        "do_sample": getattr(config, "do_sample", None),
        "temperature": getattr(config, "temperature", None),
        "warmup_iter": getattr(config, "warmup_iter", None),
        "cache_implementation": getattr(config, "cache_implementation", None),
        "compile_mode": _to_serializable(getattr(config, "compile_mode", None)),
        "vram_limit_gb": getattr(config, "vram_limit_gb", None),
        "cpu_offload_gb": getattr(config, "cpu_offload_gb", None),
        "generator_profiling": getattr(config, "generator_profiling", None),
        "profiling_verbose": getattr(config, "profiling_verbose", None),
        "print_time": getattr(config, "print_time", None),
        "print_message": getattr(config, "print_message", None),
        "log_dir": getattr(config, "log_dir", None),
        "out_dir": getattr(config, "out_dir", None),
        "detailed_analysis": getattr(config, "detailed_analysis", None),
        "nvtx_profiling": getattr(config, "nvtx_profiling", None),
        "nsys_output": getattr(config, "nsys_output", None),
        "generator_kwargs": _to_serializable(generator_kwargs),
        "draft_params": _to_serializable(draft_params),
        "recipe": _serialize_recipe(getattr(config, "recipe", None)),
    }

    return snapshot


def _load_yaml_config(path: str, _seen: frozenset[str] = frozenset()) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config. Install it with `pip install pyyaml`."
        ) from e

    resolved = os.path.abspath(path)
    if resolved in _seen:
        raise ValueError(f"Circular config 'extends' chain detected at {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top-level, got {type(data).__name__}")
    data = dict(data)

    # Optional single-parent inheritance: `extends: <path>` (relative to this
    # file's directory, or absolute) is deep-merged under the current config so
    # that near-identical variants (trace/sweep configs) share one source.
    base_ref = data.pop("extends", None)
    if base_ref is not None:
        base_path = os.path.expanduser(str(base_ref))
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(resolved), base_path)
        base_config = _load_yaml_config(base_path, _seen | {resolved})
        data = _deep_merge_dict(base_config, data)

    return data


def _resolve_existing_path(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def _normalize_compile_mode(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _apply_yaml_overrides(default_config: Dict[str, Any], yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    if not yaml_config:
        return dict(default_config)

    cfg = dict(yaml_config)
    cfg.pop("method", None)

    if "compile_mode" in cfg:
        cfg["compile_mode"] = _normalize_compile_mode(cfg.get("compile_mode"))

    # DraftParams can be specified as a dict in YAML.
    if isinstance(cfg.get("draft_params"), dict):
        from specdecodes.models.utils.utils import DraftParams

        base_dp = _draft_params_to_dict(default_config.get("draft_params"))
        merged_dp = _deep_merge_dict(base_dp, dict(cfg["draft_params"]))
        cfg["draft_params"] = DraftParams(**merged_dp)

    # generator_kwargs deep-merge.
    if isinstance(cfg.get("generator_kwargs"), dict):
        base_gk = default_config.get("generator_kwargs") or {}
        cfg["generator_kwargs"] = _deep_merge_dict(base_gk, cfg["generator_kwargs"])

    return _deep_merge_dict(default_config, cfg)


def _build_base_parser() -> argparse.ArgumentParser:
    # Important: disable allow_abbrev so Typer subcommand flags like --d/--k
    # don't get parsed as abbreviations for top-level options (e.g., --device).
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Decoding method to use (overrides YAML `method`)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file. Required. Values override method defaults; CLI args override YAML.",
    )

    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="KEY.PATH=VALUE",
        help=(
            "Override any config field, dotted for nesting; repeatable. Value is parsed "
            "as YAML (16 -> int, true -> bool, [a,b] -> list). Reaches nested config the "
            "named flags can't, e.g. --set draft_params.max_depth=8 "
            "--set generator_kwargs.verify_kwargs.threshold=0.3. Precedence: named flag "
            "> --set > YAML > method default."
        ),
    )
    return parser


def _maybe_reexec_with_nsys(enabled: bool, output: str) -> None:
    if not enabled:
        return

    # Avoid infinite recursion when we re-exec under nsys.
    if os.environ.get("SUBSPEC_NSYS_ACTIVE", "0") == "1":
        return

    if shutil.which("nsys") is None:
        print("Error: NVTX profiling requested but `nsys` was not found in PATH.")
        sys.exit(1)

    os.environ["SUBSPEC_NSYS_ACTIVE"] = "1"

    # Mirrors the previous wrapper-script settings, but lives in Python so it's config-driven.
    cmd = [
        "nsys",
        "profile",
        "-w",
        "true",
        "-t",
        "cuda,nvtx,osrt,cudnn,cublas",
        "-s",
        "cpu",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cudabacktrace=all",
        "--force-overwrite=true",
        "--python-sampling-frequency=1000",
        "--python-sampling=true",
        "--cuda-memory-usage=true",
        "--gpuctxsw=true",
        "--python-backtrace",
        "-x",
        "true",
        "-o",
        output,
        sys.executable,
        "-m",
        "run.main",
        *sys.argv[1:],
    ]
    os.execvp(cmd[0], cmd)


def _build_full_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Parser that only separates the top-level flags from the Typer subcommand argv.

    Config fields are set via YAML + ``--set key.path=value`` (see `_build_base_parser`),
    not per-field CLI flags.
    """
    return argparse.ArgumentParser(parents=[base_parser], add_help=False, allow_abbrev=False)


def _enforce_benchmark_requires_config(typer_argv: list[str], config_path: str | None) -> None:
    if typer_argv and typer_argv[0] in BENCHMARK_COMMANDS and config_path is None:
        print(
            "Error: benchmark commands require a YAML config via --config.\n"
            "Example: python -m run.main --config configs/methods/subspec_sd.yaml run-benchmark --benchmarks mt-bench --max-samples 20"
        )
        sys.exit(2)


def _resolve_method(cli_method: str | None, yaml_config: Dict[str, Any]) -> str:
    if isinstance(cli_method, str) and cli_method.strip():
        return cli_method
    if isinstance(yaml_config.get("method"), str) and yaml_config["method"].strip():
        return yaml_config["method"]
    raise ValueError("Missing `method`: specify --method or set `method:` in the YAML config.")


def _load_yaml_and_method(args: argparse.Namespace) -> tuple[str, Dict[str, Any], str]:
    try:
        config_path = _resolve_existing_path(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {os.path.abspath(os.path.expanduser(args.config))}")
        sys.exit(1)

    yaml_config = _load_yaml_config(config_path)

    try:
        method = _resolve_method(args.method, yaml_config)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    return config_path, yaml_config, method


def _effective_nsys_settings(yaml_config: Dict[str, Any]) -> tuple[bool, str]:
    # nvtx/nsys are configured via YAML or `--set` (already merged into yaml_config here).
    return (
        bool(yaml_config.get("nvtx_profiling", False)),
        str(yaml_config.get("nsys_output", "nsight_report")),
    )


def _configure_wandb_flags(config: "AppConfig") -> None:
    # Propagate global research flags via wandb_logger (avoids env var plumbing).
    try:
        from specdecodes.models.utils.wandb_logger import wandb_logger

        wandb_logger.set_flags(
            detailed_analysis=bool(getattr(config, "detailed_analysis", False)),
            nvtx_profiling=bool(getattr(config, "nvtx_profiling", False)),
        )
    except Exception:
        # Keep main robust even if wandb_logger isn't importable in some minimal setups.
        pass


def _build_app_config(
    *,
    AppConfig: type["AppConfig"],
    method: str,
    default_config: Dict[str, Any],
) -> "AppConfig":
    config = AppConfig()
    config.method = method
    config.update(default_config)
    return config


def main():
    # Configure env + compatibility patches before importing heavy GPU code.
    _configure_runtime_environment()
    _maybe_patch_auto_gptq()

    # 1) Parse method + YAML config path first to load defaults
    base_parser = _build_base_parser()
    args, _ = base_parser.parse_known_args()
    config_path, yaml_config, method = _load_yaml_and_method(args)

    # Fold `--set key.path=value` overrides in above the YAML (below named flags), so they
    # flow through the normal draft_params/generator_kwargs handling and reach any field.
    try:
        set_overrides = _parse_set_overrides(getattr(args, "set_overrides", None))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)
    if set_overrides:
        yaml_config = _deep_merge_dict(yaml_config, set_overrides)

    # If enabled via YAML/CLI, re-exec under Nsight Systems *before* importing heavy GPU code.
    nsys_enabled, nsys_output = _effective_nsys_settings(yaml_config)
    _maybe_reexec_with_nsys(nsys_enabled, nsys_output)

    # Import project modules lazily so env defaults above apply before any torch/CUDA init.
    from .core.configuration import AppConfig
    from .core.registry import ModelRegistry
    from .core.presets import register_presets
    from .core.builder import GeneratorPipelineBuilder
    from .core.router import run_app
    from .core.config_utils import instantiate_recipe

    # 2) Register presets (after optional nsys re-exec)
    register_presets()
    
    # 3) Get default config for the method
    method_entry = ModelRegistry.get(method)
    if method_entry is None:
        print(f"Unknown method: {method}. Available methods: {ModelRegistry.list_methods()}")
        sys.exit(1)
        
    default_config = method_entry.default_config.copy()

    # Merge YAML into default_config (defaults <- yaml).
    default_config = _apply_yaml_overrides(default_config, yaml_config)
    
    # 4) Build full parser for AppConfig (method defaults <- YAML; CLI overrides both)
    full_parser = _build_full_parser(base_parser)
    
    # Parse again with known args to override defaults
    # We still use parse_known_args because run_app (Typer) needs the rest
    _, typer_argv = full_parser.parse_known_args()

    # (Kept for backward compatibility + explicit error messaging if this file is reused elsewhere.)
    _enforce_benchmark_requires_config(typer_argv, args.config)
    
    # 5) Build AppConfig (unknown keys from YAML / --set raise here)
    try:
        config = _build_app_config(
            AppConfig=AppConfig,
            method=method,
            default_config=default_config,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)
    _configure_wandb_flags(config)

    # Allow YAML to specify recipes via import path + kwargs.
    config.recipe = instantiate_recipe(getattr(config, "recipe", None))
    config.config_path = config_path
    config.settings_snapshot = _build_settings_snapshot(
        config=config,
        config_path=config_path,
        subcommand_argv=typer_argv,
    )
    
    # 6. Build pipeline
    # We must patch sys.argv for Typer to work correctly on the subcommands
    # Typer expects [script, subcommand, options...]
    # We removed the config options, so we pass the rest.
    sys.argv = [sys.argv[0]] + typer_argv
    
    builder = GeneratorPipelineBuilder(config)
    run_app(builder)

if __name__ == "__main__":
    main()
