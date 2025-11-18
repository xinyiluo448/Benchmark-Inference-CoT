from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline


def load_model_and_tokenizer(model_name: str, load_4bit: bool = False) -> Tuple[TextGenerationPipeline, AutoTokenizer]:
    quantization_config = None
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("bitsandbytes not installed; cannot use --load-4bit") from exc
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {
        "torch_dtype": torch.float16,
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
