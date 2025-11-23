from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class KVStrategy:
    raw: str
    label: str
    use_cache: bool
    sliding_window: Optional[int] = None
    cache_impl: Optional[str] = None


def parse_kv_strategy(value: Optional[str]) -> KVStrategy:
    if not value:
        value = "default"
    raw = value
    val = value.strip().lower()
    if val in {"default", "on", "enable", "enabled", "true"}:
        return KVStrategy(raw, "default", True)
    if val in {"off", "disable", "disabled", "nocache", "false"}:
        return KVStrategy(raw, "off", False)
    if val in {"dynamic", "static", "quantized"}:
        return KVStrategy(raw, val, True, cache_impl=val)
    sliding_match = re.match(r"(sliding|window)([:=]?)(\d+)?", val)
    if sliding_match and sliding_match.group(3):
        window = int(sliding_match.group(3))
        return KVStrategy(raw, f"sliding{window}", True, sliding_window=window)
    if val.startswith("sliding") and ":" in val:
        try:
            window = int(val.split(":", 1)[1])
            return KVStrategy(raw, f"sliding{window}", True, sliding_window=window)
        except ValueError:
            pass
    label = val.replace(" ", "_") or "custom"
    return KVStrategy(raw, label, True)


def apply_kv_strategy(model, strategy: KVStrategy):
    if hasattr(model, "config"):
        model.config.use_cache = strategy.use_cache
        if strategy.sliding_window is not None:
            setattr(model.config, "sliding_window", strategy.sliding_window)
        if strategy.cache_impl:
            setattr(model.config, "cache_implementation", strategy.cache_impl)
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = strategy.use_cache
        if strategy.cache_impl:
            setattr(model.generation_config, "cache_implementation", strategy.cache_impl)
