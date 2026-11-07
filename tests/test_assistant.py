"""Offline tests: they use FakeProvider, so no API key or network is needed."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import ROOT, Settings
from app.knowledge import KnowledgeBase, load_chunks, split_markdown
from app.providers import FakeProvider
from app.rag import RAGService


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(gemini_api_key=None, chroma_dir=tmp_path / "chroma", rate_limit_per_minute=3)


@pytest.fixture
def kb(settings: Settings) -> KnowledgeBase:
    kb = KnowledgeBase(settings.chroma_dir, settings.collection)
    kb.build(load_chunks(ROOT / "knowledge"), FakeProvider())
    return kb


# ---------------------------------------------------------------- knowledge base
def test_markdown_is_split_by_section():
    chunks = split_markdown(ROOT / "knowledge" / "trainers.md")
    sections = [c.section for c in chunks]
    assert "Trainers" in sections and "Which trainer should I choose?" in sections
    assert all(c.text.startswith("Trainers") for c in chunks), "document title is kept in every chunk"


def test_chunk_ids_are_stable():
    assert [c.id for c in load_chunks(ROOT / "knowledge")] == [c.id for c in load_chunks(ROOT / "knowledge")]


def test_rebuild_is_skipped_when_nothing_changed(kb: KnowledgeBase):
    assert kb.build(load_chunks(ROOT / "knowledge"), FakeProvider()) == 0


# ---------------------------------------------------------------- retrieval + answering
def test_retrieval_finds_the_right_section(kb: KnowledgeBase):
    rag = RAGService(kb, FakeProvider(), top_k=3, min_similarity=0.0)
    files = [h["file"] for h in rag.retrieve("Which trainer teaches boxing?")]
    assert "trainers.md" in files


def test_sources_are_wrapped_as_data(kb: KnowledgeBase):
    provider = FakeProvider()
    RAGService(kb, provider, min_similarity=0.0).answer("How much is the yearly plan?")
    prompt = provider.calls[-1]["prompt"]
    assert "<sources>" in prompt and "<question>" in prompt


def test_no_relevant_sources_means_not_grounded(kb: KnowledgeBase):
    rag = RAGService(kb, FakeProvider(), min_similarity=0.99)  # nothing passes the threshold
    result = rag.answer("What's the weather in Istanbul?")
    assert result.grounded is False and result.sources == []


# ---------------------------------------------------------------- API
def make_client(settings: Settings, provider=None) -> TestClient:
    return TestClient(create_app(settings, provider))


def test_health_and_ask(settings: Settings):
    with make_client(settings, FakeProvider()) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ask", json={"question": "Who is the calisthenics trainer?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"answer", "language", "grounded", "sources", "latency_ms"}


def test_input_validation(settings: Settings):
    with make_client(settings, FakeProvider()) as client:
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
        assert client.post("/ask", json={}).status_code == 422


def test_rate_limit(settings: Settings):
    with make_client(settings, FakeProvider()) as client:
        codes = [client.post("/ask", json={"question": f"plans {i}"}).status_code for i in range(4)]
        assert codes == [200, 200, 200, 429]


def test_missing_api_key_gives_503_not_a_crash(settings: Settings):
    with make_client(settings) as client:  # no provider and no key
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/ask", json={"question": "hello"}).status_code == 503
