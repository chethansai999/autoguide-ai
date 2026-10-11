"""Optional local MLX generation through the existing LLMClient contract."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.config import Settings
from core.llm import LLMResult

log = logging.getLogger(__name__)
_MODEL_LOCK = threading.Lock()
_MODELS: dict[str, tuple[Any, Any]] = {}


def model_artifacts_present(directory: Path) -> bool:
    """Inspect files only; do not import the optional runtime or load weights."""
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    if not all((directory / name).is_file() for name in required):
        return False
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            return bool(shards) and all(
                Path(name).name == name and (directory / name).is_file() for name in shards
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return any(directory.glob("*.safetensors"))


class LocalMLXClient:
    provider = "local"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.local_model
        self.path = settings.local_model_path.expanduser().resolve()

    def complete(self, system, messages, *, temperature=None, max_tokens=None, json_mode=False):
        start = time.perf_counter()
        if not model_artifacts_present(self.path):
            raise ValueError("Local model files are missing or incomplete. Configure LOCAL_MODEL_PATH in .env.")
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as error:
            raise RuntimeError("Install requirements-local.txt on Apple Silicon to use LLM_PROVIDER=local.") from error

        # A single API process shares the model, including across independently
        # constructed clients. Serial GPU access avoids duplicate allocations.
        with _MODEL_LOCK:
            key = str(self.path)
            if key not in _MODELS:
                log.info("Loading local generation model: %s", self.model)
                _MODELS[key] = load(key, tokenizer_config={"trust_remote_code": False, "local_files_only": True})
            model, tokenizer = _MODELS[key]
            if json_mode:
                system += "\n\nRespond with a single JSON object and nothing else."
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, *messages],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            output_limit = max_tokens if max_tokens is not None else self.settings.llm_max_tokens
            if output_limit <= 0 or prompt_tokens + output_limit > self.settings.local_context_tokens:
                raise ValueError("Local model context budget exceeded. Start a new conversation or increase LOCAL_CONTEXT_TOKENS.")
            sampler = make_sampler(temp=self.settings.llm_temperature if temperature is None else temperature)
            text = generate(model, tokenizer, prompt=prompt, max_tokens=output_limit,
                            sampler=sampler, prefill_step_size=512, verbose=False).strip()
            if not text:
                raise RuntimeError("Local model returned an empty answer.")
            completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        return LLMResult(text=text, provider=self.provider, model=self.model,
                         prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                         latency_ms=(time.perf_counter() - start) * 1000, cost_usd=0.0)
