from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline


def _resolve_dtype(dtype_str: str):
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.float16)


def load_model_and_tokenizer(
    model_name: str,
    load_4bit: bool = False,
    bits: Optional[int] = None,
    dtype: str = "float16",
) -> Tuple[TextGenerationPipeline, AutoTokenizer]:
    quantization_config = None
    target_bits = bits if bits is not None else (4 if load_4bit else None)
    if target_bits in (4, 8):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("bitsandbytes not installed; cannot use quantized loading") from exc
        if target_bits == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {
        "torch_dtype": _resolve_dtype(dtype),
        "device_map": "auto",
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # Ensure pad_token_id is set for generation
    if model.config.pad_token_id is None and tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    return pipeline, tokenizer
