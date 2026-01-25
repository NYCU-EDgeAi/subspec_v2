"""Monkey patching utilities for FlashInfer v3.

Direct copy of v1 implementation (known working).
"""

from typing import Callable
from .rms_norm import FiLlamaRMSNorm
from .attention import FiLlamaAttention, FiQwen3Attention
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    """Binds a new method to a module instance."""
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(module, eps=1e-6):
    """Patch RMSNorm module with FlashInfer implementation."""
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    _bind_method_to_module(module, "forward", FiLlamaRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", FiLlamaRMSNorm.extra_repr)


def _patch_attention_module(module, use_ragged=False):
    """Patch attention module with FlashInfer implementation."""
    if use_ragged:
        pass
    else:
        if isinstance(module, LlamaAttention):
            _bind_method_to_module(module, "forward", FiLlamaAttention.forward)
        elif isinstance(module, Qwen3Attention):
            _bind_method_to_module(module, "forward", FiQwen3Attention.forward)
        else:
            raise ValueError(f"Unsupported attention module type: {type(module)}")


def apply_flashinfer_kernel_to_llama(
    attention: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_ragged: bool = False,
) -> None:
    """Apply FlashInfer kernels to HuggingFace Llama/Qwen models.

    Args:
        attention: Whether to apply FlashInfer attention. Default is True.
        rms_norm: Whether to apply FlashInfer RMSNorm. Default is True.
        swiglu: Whether to apply SwiGLU MLP. Default is True.
        model: The model instance to patch. Default is None.
        use_ragged: Whether to use ragged attention. Default is False.
    """
    from transformers.models.llama import modeling_llama
    from transformers.models.qwen3 import modeling_qwen3

    if rms_norm:
        modeling_llama.LlamaRMSNorm = FiLlamaRMSNorm
        modeling_qwen3.Qwen3RMSNorm = FiLlamaRMSNorm

    if attention:
        if use_ragged:
            pass
        else:
            modeling_llama.LlamaAttention = FiLlamaAttention
            modeling_qwen3.Qwen3Attention = FiQwen3Attention

    if model is not None:
        # For target model
        if hasattr(model, "base_model_prefix"):
            base_model = getattr(model, model.base_model_prefix, model)
        else:
            # Fallback for draft models
            base_model = getattr(model, "model", model).model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
            if attention:
                _patch_attention_module(decoder_layer.self_attn, use_ragged)
