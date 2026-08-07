import difflib
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, Union
import torch
from specdecodes.models.utils.utils import DraftParams

@dataclass
class AppConfig:
    # Base configurations
    method: str = "classic_sd"
    # Attention/KV-cache backend the generator drives. "sdpa" (static/dynamic Cache)
    # or "flashinfer" (paged RequestKvCache). Selects the SpecDecodeBackend adapter for
    # methods that support the seam; FI registry entries default it to "flashinfer".
    backend: str = "sdpa"
    vram_limit_gb: Optional[int] = None
    seed: int = 0
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    
    # Model paths
    llm_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model_path: Optional[str] = None
    
    # Generation parameters
    max_length: int = 2048
    do_sample: bool = False
    temperature: float = 0.0
    
    # Generator-specific configurations
    generator_kwargs: Dict[str, Any] = field(default_factory=dict)
    draft_params: Optional[DraftParams] = None
    
    # Recipe
    recipe: Any = None
    cpu_offload_gb: Optional[int] = None
    
    # Additional configurations
    cache_implementation: str = "dynamic"
    warmup_iter: int = 0
    compile_mode: Optional[str] = None
    
    # Profiling
    generator_profiling: bool = True
    profiling_verbose: bool = True
    print_time: bool = True
    print_message: bool = True
    
    # Benchmarking/logging
    out_dir: Optional[str] = None
    log_dir: str = "experiments"

    # Settings snapshot (resolved config + CLI context)
    config_path: Optional[str] = None
    settings_snapshot: Optional[Dict[str, Any]] = None

    # Research toggles (set via YAML/CLI)
    detailed_analysis: bool = False
    nvtx_profiling: bool = False
    nsys_output: str = "nsight_report"

    def update(self, new_config: Dict[str, Any]):
        """Set config fields from a dict, rejecting unknown keys.

        Previously unknown keys were silently dropped, so a typo like `max_lenght`
        or `draft_parms` became a no-op that quietly ran the default. Now they raise
        with a did-you-mean hint (every field a config may set is an AppConfig field;
        `extends` is consumed earlier at the YAML layer)."""
        known = {f.name for f in fields(self)}
        unknown = [k for k in new_config if k not in known]
        if unknown:
            hints = []
            for k in unknown:
                near = difflib.get_close_matches(str(k), known, n=1)
                hints.append(repr(k) + (f" (did you mean {near[0]!r}?)" if near else ""))
            raise ValueError(
                "Unknown config key(s): "
                + ", ".join(hints)
                + f".\nKnown keys: {', '.join(sorted(known))}."
            )
        for key, value in new_config.items():
            setattr(self, key, value)
