"""Local provider integration, entirely offline with simulated MLX modules."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.config import Settings
from core.llm import build_llm, is_connection_error
from core import local_llm


@pytest.fixture
def local_settings(tmp_path):
    directory = tmp_path / "local-model"
    directory.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (directory / name).write_text("{}")
    (directory / "model.safetensors").write_bytes(b"synthetic weights; never loaded")
    return Settings(_env_file=None, llm_provider="local", local_model_path=directory,
                    local_model="offline-fixture", llm_max_tokens=50, local_context_tokens=1024,
                    openai_api_key="", anthropic_api_key="", gemini_api_key="")


@pytest.fixture
def runtime(monkeypatch):
    seen = {"loads": [], "generations": []}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["template_options"] = kwargs
            return " ".join(message["content"] for message in messages)

        def encode(self, text, **_kwargs):
            return text.split()

    def load(path, **kwargs):
        seen["loads"].append((path, kwargs))
        return object(), Tokenizer()

    def generate(model, tokenizer, **kwargs):
        seen["generations"].append(kwargs)
        return "A cited fixture answer [1]."

    monkeypatch.setattr(local_llm, "_MODELS", {})
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load, generate=generate))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=lambda **kw: kw))
    return seen


def test_local_registry_is_lazy_and_requires_no_api_key(local_settings, runtime):
    client = build_llm(local_settings)
    assert isinstance(client, local_llm.LocalMLXClient)
    assert runtime["loads"] == []
    assert local_settings.resolved_llm_provider() == "local"


def test_local_request_preserves_instructions_history_and_usage(local_settings, runtime):
    result = build_llm(local_settings).complete("Cite evidence.", [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Earlier answer [1]."},
        {"role": "user", "content": "Follow up"},
    ], max_tokens=30, temperature=0)
    assert runtime["messages"][0] == {"role": "system", "content": "Cite evidence."}
    assert [m["role"] for m in runtime["messages"]] == ["system", "user", "assistant", "user"]
    request = runtime["generations"][0]
    assert request["sampler"] == {"temp": 0}
    assert request["max_tokens"] == 30 and request["prefill_step_size"] == 512
    assert runtime["loads"][0][1]["tokenizer_config"] == {"trust_remote_code": False, "local_files_only": True}
    assert result.provider == "local" and result.model == "offline-fixture"
    assert result.prompt_tokens > 0 and result.completion_tokens == 5
    assert result.cost_usd == 0 and result.latency_ms > 0
    assert result.as_trace()["provider"] == "local"


def test_model_reused_across_clients_and_json_instruction_retained(local_settings, runtime):
    build_llm(local_settings).complete("s", [{"role": "user", "content": "x"}])
    build_llm(local_settings).complete("s", [{"role": "user", "content": "x"}], json_mode=True)
    assert len(runtime["loads"]) == 1
    assert "single JSON object" in runtime["messages"][0]["content"]


def test_missing_weights_fail_without_load_or_download(local_settings, runtime):
    (local_settings.local_model_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or incomplete"):
        build_llm(local_settings).complete("s", [{"role": "user", "content": "x"}])
    assert runtime["loads"] == [] and runtime["generations"] == []


def test_incomplete_sharded_model_is_not_ready(local_settings):
    directory = local_settings.local_model_path
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.one": "model.safetensors", "layer.two": "missing-shard.safetensors"}}))
    assert not local_llm.model_artifacts_present(directory)


def test_context_overflow_rejects_instead_of_truncating_evidence(local_settings, runtime):
    local_settings.local_context_tokens = 4
    with pytest.raises(ValueError, match="context budget exceeded"):
        build_llm(local_settings).complete("Keep all evidence", [{"role": "user", "content": "x"}])
    assert runtime["generations"] == []


@pytest.mark.parametrize("empty", [False, True])
def test_generation_errors_are_visible(local_settings, runtime, monkeypatch, empty):
    def generate(*_args, **_kwargs):
        if empty:
            return ""
        raise RuntimeError("Local inference failed")

    monkeypatch.setattr(sys.modules["mlx_lm"], "generate", generate)
    with pytest.raises(RuntimeError) as caught:
        build_llm(local_settings).complete("s", [{"role": "user", "content": "x"}])
    assert not is_connection_error(caught.value)


@pytest.mark.parametrize("missing", [None, "runtime", "weights"])
def test_local_readiness_checks_files_without_loading(local_settings, runtime, monkeypatch, missing):
    from api import readiness

    settings = local_settings
    settings.data_dir = settings.local_model_path.parent / "data"
    settings.embedding_backend = "sentence-transformers"
    path = settings.chunks_dir() / "chunks.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"fixture-id","text":"Fixture evidence."}\n')
    registry = SimpleNamespace(collection=settings.collection_name(), strategy=settings.chunk_strategy,
        version=settings.chunk_version, embedding_model=settings.embedding_model, chunk_count=1)
    factory = MagicMock()
    db = factory.return_value.__enter__.return_value
    db.scalar.return_value = 1
    db.scalars.return_value.all.return_value = [registry]
    monkeypatch.setattr(readiness, "_vector_ids", lambda *_args: {"fixture-id"})
    monkeypatch.setattr(readiness, "find_spec", lambda _name: None if missing == "runtime" else object())
    if missing == "weights":
        (settings.local_model_path / "model.safetensors").unlink()
    report = readiness.inspect_readiness(SimpleNamespace(settings=settings, session_factory=factory))
    assert report["checks"]["generation_configured"] is (missing is None)
    assert report["ready"] is (missing is None)
    assert report["llm_provider"] == "local" and not report["demo"]
    assert runtime["loads"] == [], "readiness must not load MLX weights"


def test_unknown_availability_cannot_be_rewritten_as_absence():
    from core.schemas import Evidence
    from rag.generation import GroundedGenerator

    class MustNotGenerate:
        def complete(self, *_args, **_kwargs):
            pytest.fail("Unknown structured availability must not become a model's yes/no guess")

    evidence = [Evidence(ref=1, evidence_id="vehicle:fixture", kind="vehicle_record",
        title="Fixture vehicle", url="https://www.nhtsa.gov/ratings",
        text="Vehicle: 2020 TOYOTA CAMRY\nRearview (backup) camera: not reported\nBlind spot detection: optional")]
    answer = GroundedGenerator(llm=MustNotGenerate()).generate(
        "Does the 2020 Toyota Camry have a backup camera and blind spot detection?", evidence)
    assert answer.valid_citations and answer.cited_refs == {1}
    assert "not reported" in answer.text and "availability is unknown" in answer.text
    assert "blind spot detection optional" in answer.text
    assert "does not have" not in answer.text
    assert answer.llm.provider == "structured-data" and answer.llm.model == "none"
    assert "unknown_feature_availability_preserved" in answer.flags


def test_explicit_unavailability_remains_distinct_from_unknown():
    from core.schemas import Evidence
    from rag.generation import unknown_feature_answer

    evidence = [Evidence(ref=1, evidence_id="vehicle:fixture", kind="vehicle_record",
        title="Fixture vehicle", url="https://www.nhtsa.gov/ratings",
        text="Vehicle: 2020 TOYOTA CAMRY\nRearview (backup) camera: not available")]
    assert unknown_feature_answer("Does it have a backup camera?", evidence) is None
