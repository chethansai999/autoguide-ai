"""Provider-agnostic LLM access.

Every LLM call in the project goes through ``get_llm().complete(...)`` so that
(a) the team can develop without a key (MockLLM), (b) the provider can be
swapped by env var, and (c) tokens / latency / cost are recorded uniformly
for the cost-per-query metric.

    LLM_PROVIDER=auto      # openai if OPENAI_API_KEY set, else anthropic, else mock
    LLM_PROVIDER=openai    OPENAI_API_KEY=...  OPENAI_MODEL=gpt-4o-mini
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=...
    LLM_PROVIDER=gemini    GEMINI_API_KEY=...    GEMINI_MODEL=gemini-3.8-flash
    LLM_PROVIDER=local     # optional MLX runtime, existing local model, no API key
    LLM_PROVIDER=mock      # deterministic extractive answers, no network
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from core.config import Settings, get_settings
from core.netcheck import ensure_reachable, mark_down

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    raw: dict = field(default_factory=dict)
    # Set when the configured provider was unreachable and the answer was
    # built extractively from the evidence instead (see is_connection_error).
    fallback_reason: str | None = None

    def as_trace(self) -> dict:
        trace = {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": self.cost_usd,
        }
        if self.fallback_reason:
            trace["fallback_reason"] = self.fallback_reason
        return trace


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult: ...


# Exception class names (from openai, anthropic, httpx, httpcore and the
# standard library) that mean "the request never got a response". Matched by
# name so this module does not import optional SDKs.
_CONNECTION_ERROR_NAMES = {
    "APIConnectionError", "APITimeoutError",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "RemoteProtocolError", "TimeoutException", "NetworkError",
    "gaierror", "ConnectionError", "TimeoutError",
}


def is_connection_error(error: BaseException) -> bool:
    """True for network failures (DNS, refused, reset, timeout). False for
    errors the provider actually returned, such as a bad key, an exhausted
    quota or a malformed request: those must stay visible."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "status_code", None) is not None:
            return False  # the provider answered (401, 429, 500...): not a network failure
        if any(cls.__name__ in _CONNECTION_ERROR_NAMES for cls in type(current).__mro__):
            return True
        # Follow explicit ``raise ... from`` links only. Implicit context could
        # attach an earlier, retried connection error to a later 401.
        current = current.__cause__
    return False


def _timeout(settings: Settings):
    """Whole-request timeout, with a shorter connect phase."""
    import httpx

    return httpx.Timeout(settings.llm_timeout_s, connect=min(10.0, settings.llm_timeout_s))


@contextmanager
def _network_guard(host: str):
    """Skip a host that is known to be down; remember new network failures."""
    ensure_reachable(host)
    try:
        yield
    except Exception as error:
        if is_connection_error(error):
            mark_down(host)
        raise


def _cost(settings: Settings, prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens / 1e6 * settings.llm_price_input_per_mtok
        + completion_tokens / 1e6 * settings.llm_price_output_per_mtok,
        6,
    )


class OpenAIClient:
    provider = "openai"

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.settings = settings
        self.model = settings.openai_model
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=_timeout(settings),
            max_retries=settings.llm_max_retries,
        )
        self.host = self.client.base_url.host

    def complete(self, system, messages, *, temperature=None, max_tokens=None, json_mode=False):
        start = time.perf_counter()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with _network_guard(self.host):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                messages=[{"role": "system", "content": system}, *messages],
                **kwargs,
            )
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return LLMResult(
            text=(response.choices[0].message.content or "").strip(),
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=_cost(self.settings, prompt_tokens, completion_tokens),
        )


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, settings: Settings):
        import anthropic  # optional dependency: pip install anthropic

        self.settings = settings
        self.model = settings.anthropic_model
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=_timeout(settings),
            max_retries=settings.llm_max_retries,
        )
        self.host = self.client.base_url.host

    def complete(self, system, messages, *, temperature=None, max_tokens=None, json_mode=False):
        start = time.perf_counter()
        if json_mode:
            system = system + "\n\nRespond with a single JSON object and nothing else."
        with _network_guard(self.host):
            response = self.client.messages.create(
                model=self.model,
                system=system,
                temperature=self.settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                messages=messages,
            )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=_cost(self.settings, prompt_tokens, completion_tokens),
        )


class GeminiAPIError(RuntimeError):
    """An error the Gemini API returned (bad key, quota, unknown model...).

    Carries ``status_code`` so is_connection_error() never treats it as a
    network failure: provider errors must stay visible.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(f"Gemini API {status_code}: {message}")
        self.status_code = status_code


class GeminiClient:
    """Google Gemini through the native REST API.

    Uses the documented ``x-goog-api-key`` header, which also works for the
    newer "AQ." AI Studio keys (some are rejected as Bearer tokens on the
    OpenAI-compatible endpoint). Only httpx is needed.
    """

    provider = "gemini"

    def __init__(self, settings: Settings, transport=None):
        import httpx

        self.settings = settings
        self.model = settings.gemini_model
        self.base = settings.gemini_api_base.rstrip("/")
        self.host = httpx.URL(self.base).host
        self.http = httpx.Client(
            timeout=_timeout(settings),
            transport=transport or httpx.HTTPTransport(retries=settings.llm_max_retries),
            headers={"x-goog-api-key": settings.gemini_api_key or "", "content-type": "application/json"},
        )

    def complete(self, system, messages, *, temperature=None, max_tokens=None, json_mode=False):
        start = time.perf_counter()
        config: dict = {
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            # Newer Gemini models spend part of this budget on internal reasoning,
            # so leave headroom; prompts keep the visible answer short.
            "maxOutputTokens": max(4 * (max_tokens or self.settings.llm_max_tokens), 2048),
        }
        if json_mode:
            config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": config,
        }
        with _network_guard(self.host):
            response = self.http.post(f"{self.base}/models/{self.model}:generateContent", json=body)
        if response.status_code >= 400:
            try:
                message = response.json().get("error", {}).get("message", "")
            except ValueError:
                message = response.text
            raise GeminiAPIError(response.status_code, " ".join(str(message).split())[:300])
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            raise GeminiAPIError(200, f"empty response ({reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text.strip():
            raise GeminiAPIError(200, f"empty answer (finishReason={candidates[0].get('finishReason')})")
        usage = data.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount") or 0)
        completion_tokens = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=_cost(self.settings, prompt_tokens, completion_tokens),
        )


_EVIDENCE_BLOCK = re.compile(
    r"Evidence \[(\d+)\].*?Content:\s*(.*?)(?=\n<<<END EVIDENCE|\Z)", re.S
)


class MockLLM:
    """Deterministic, offline stand-in for development and tests.

    Produces a short *extractive* answer from the first evidence blocks in
    the prompt and cites them, so the full pipeline (citations, validation,
    UI) can be exercised before an API key exists. It is NOT a quality
    baseline - never report its outputs as results.
    """

    provider = "mock"
    model = "mock-extractive-v1"
    NOTE = "(Mock answer - set an API key for real generation.)"

    def __init__(self, note: str | None = None):
        self.note = self.NOTE if note is None else note

    def complete(self, system, messages, *, temperature=None, max_tokens=None, json_mode=False):
        start = time.perf_counter()
        prompt = messages[-1]["content"] if messages else ""
        if json_mode:
            text = "{}"
        else:
            blocks = _EVIDENCE_BLOCK.findall(prompt)
            if not blocks:
                text = "I don't have enough information in the available documents to answer that."
            else:
                sentences = []
                for ref, content in blocks[:2]:
                    lines = [
                        re.sub(r"^#+\s*", "", l.strip()) for l in content.splitlines()
                        if l.strip() and not l.strip().startswith("[removed")
                    ]
                    lines = [l for l in lines if len(l.split()) > 3 or ":" in l]
                    if lines and ":" in lines[0]:  # structured record: first facts
                        first = "; ".join(lines[1:5] if len(lines) > 1 else lines[:1])
                    else:
                        first = re.split(r"(?<=[.!?])\s+", " ".join(" ".join(lines).split()))[0]
                    words = first.split()
                    if len(words) > 20:  # stay under the 25-word quote limit
                        first = " ".join(words[:20]) + "..."
                    sentences.append(f"{first} [{ref}]")
                text = " ".join([self.note, *sentences]).strip()
        prompt_tokens = len(prompt.split())
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=0.0,
        )


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.resolved_llm_provider()
    if provider == "openai":
        return OpenAIClient(settings)
    if provider == "anthropic":
        return AnthropicClient(settings)
    if provider == "gemini":
        return GeminiClient(settings)
    if provider == "local":
        from core.local_llm import LocalMLXClient

        return LocalMLXClient(settings)
    log.warning("LLM provider is 'mock': answers are extractive placeholders.")
    return MockLLM()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return build_llm()
