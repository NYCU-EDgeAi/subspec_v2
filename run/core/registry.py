from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Callable, Dict, Optional


def _resolve_symbol(spec: Any, *, method_name: str, role: str) -> Any:
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec

    module_name, sep, symbol_name = spec.partition(":")
    if not sep or not module_name or not symbol_name:
        raise ValueError(
            f"Invalid lazy import spec for method '{method_name}' ({role}): '{spec}'. "
            "Expected format 'some.module:SymbolName'."
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # ImportError, OSError, etc
        raise ImportError(
            f"Failed to import module '{module_name}' for method '{method_name}' ({role})."
        ) from e

    try:
        return getattr(module, symbol_name)
    except AttributeError as e:
        raise ImportError(
            f"Failed to resolve symbol '{symbol_name}' from module '{module_name}' "
            f"for method '{method_name}' ({role})."
        ) from e


@dataclass
class ModelRegistryEntry:
    name: str
    generator_cls: Any
    draft_model_cls: Any
    default_config: Dict[str, Any]
    load_model_fn: Optional[Callable] = None
    load_draft_model_fn: Optional[Callable] = None
    load_kv_cache_fn: Optional[Callable] = None
    # Whether the method needs a separate draft KV cache when a draft model is present.
    # Many methods (e.g., SubSpec) can share the target KV cache.
    needs_draft_kv_cache: bool = True
    # Whether to compile target model forward for this method.
    compile_target: bool = True
    # Per-backend field overrides, keyed by the `backend:` config value. Lets one
    # method entry serve several attention backends (e.g. SDPA + FlashInfer) instead
    # of registering a twin per backend; `for_backend()` applies the overrides.
    backends: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    _resolved_generator_cls: Any = field(default=None, init=False, repr=False, compare=False)
    _resolved_draft_model_cls: Any = field(default=None, init=False, repr=False, compare=False)

    #: Fields a per-backend override block may replace.
    _BACKEND_OVERRIDABLE = (
        "generator_cls",
        "draft_model_cls",
        "load_model_fn",
        "load_draft_model_fn",
        "load_kv_cache_fn",
        "needs_draft_kv_cache",
        "compile_target",
    )

    def for_backend(self, backend: Any) -> "ModelRegistryEntry":
        """Return this entry with its `backends[backend]` overrides applied.

        Returns ``self`` unchanged when the backend has no override block, so callers
        can use it unconditionally. Unknown override keys raise (typo guard)."""
        import dataclasses

        overrides = self.backends.get(str(backend))
        if not overrides:
            return self
        bad = set(overrides) - set(self._BACKEND_OVERRIDABLE)
        if bad:
            raise ValueError(
                f"Unknown per-backend override key(s) for method '{self.name}' "
                f"backend '{backend}': {sorted(bad)}. "
                f"Allowed: {sorted(self._BACKEND_OVERRIDABLE)}."
            )
        return dataclasses.replace(self, backends={}, **overrides)

    def get_generator_cls(self) -> Any:
        if self._resolved_generator_cls is None:
            self._resolved_generator_cls = _resolve_symbol(
                self.generator_cls,
                method_name=self.name,
                role="generator_cls",
            )
        return self._resolved_generator_cls

    def get_draft_model_cls(self) -> Any:
        if self._resolved_draft_model_cls is None:
            self._resolved_draft_model_cls = _resolve_symbol(
                self.draft_model_cls,
                method_name=self.name,
                role="draft_model_cls",
            )
        return self._resolved_draft_model_cls


class ModelRegistry:
    _registry: Dict[str, ModelRegistryEntry] = {}

    @classmethod
    def register(
        cls,
        name: str,
        generator_cls: Any,
        draft_model_cls: Any,
        default_config: Optional[Dict[str, Any]] = None,
        load_model_fn: Optional[Callable] = None,
        load_draft_model_fn: Optional[Callable] = None,
        load_kv_cache_fn: Optional[Callable] = None,
        needs_draft_kv_cache: bool = True,
        compile_target: bool = True,
        backends: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if default_config is None:
            default_config = {}
        cls._registry[name] = ModelRegistryEntry(
            name=name,
            generator_cls=generator_cls,
            draft_model_cls=draft_model_cls,
            default_config=default_config,
            load_model_fn=load_model_fn,
            load_draft_model_fn=load_draft_model_fn,
            load_kv_cache_fn=load_kv_cache_fn,
            needs_draft_kv_cache=bool(needs_draft_kv_cache),
            compile_target=bool(compile_target),
            backends=dict(backends or {}),
        )

    @classmethod
    def get(cls, name: str) -> Optional[ModelRegistryEntry]:
        return cls._registry.get(name)

    @classmethod
    def list_methods(cls) -> list[str]:
        return sorted(cls._registry.keys())
