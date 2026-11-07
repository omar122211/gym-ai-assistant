"""AI provider layer. The rest of the app only talks to the `AIProvider` interface,
so switching from Gemini to another provider (or a local model) means adding one class here."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Kind = Literal["document", "query"]


class AIProviderError(RuntimeError):
    """The AI service failed (bad key, quota, network). Safe to show a generic message."""


class AIProvider(Protocol):
    name: str

    def embed(self, texts: list[str], *, kind: Kind) -> list[list[float]]: ...

    def generate(self, *, system: str, prompt: str, schema: type[T], temperature: float = 0.2) -> T: ...


# --------------------------------------------------------------------------- Gemini
class GeminiProvider:
    """Google Gemini API (google-genai SDK)."""

    name = "gemini"
    _RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, chat_model: str, embed_model: str, embed_dim: int, max_retries: int = 4):
        from google import genai  # imported here so tests don't need the SDK configured

        self._genai_types = __import__("google.genai.types", fromlist=["types"])
        self._errors = __import__("google.genai.errors", fromlist=["errors"])
        self.client = genai.Client(api_key=api_key)
        self.chat_model, self.embed_model, self.embed_dim = chat_model, embed_model, embed_dim
        self.max_retries = max_retries

    def _call(self, fn, *args, **kwargs):
        """Retry on rate limits / temporary errors with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except self._errors.APIError as e:
                code = getattr(e, "code", None)
                if code in self._RETRYABLE and attempt < self.max_retries:
                    wait = min(2 ** attempt * 2, 30)
                    log.warning("gemini retry", extra={"fields": {"code": code, "wait_s": wait, "attempt": attempt + 1}})
                    time.sleep(wait)
                    continue
                raise AIProviderError(f"Gemini API error {code}") from e
        raise AIProviderError("Gemini API: retries exhausted")

    def embed(self, texts: list[str], *, kind: Kind) -> list[list[float]]:
        types = self._genai_types
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            if self.embed_model.startswith("gemini-embedding-2"):
                # This model takes the task inside the text; one Content per input = one vector per input
                prefix = "task: search result | query: " if kind == "query" else "title: none | text: "
                contents = [types.Content(parts=[types.Part(text=prefix + t)]) for t in batch]
                config = types.EmbedContentConfig(output_dimensionality=self.embed_dim)
            else:
                contents = batch
                config = types.EmbedContentConfig(
                    output_dimensionality=self.embed_dim,
                    task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT",
                )
            result = self._call(self.client.models.embed_content, model=self.embed_model, contents=contents, config=config)
            if len(result.embeddings) != len(batch):
                raise AIProviderError(f"expected {len(batch)} embeddings, got {len(result.embeddings)}")
            vectors.extend(_normalize(e.values) for e in result.embeddings)
        return vectors

    def generate(self, *, system: str, prompt: str, schema: type[T], temperature: float = 0.2) -> T:
        types = self._genai_types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._call(self.client.models.generate_content, model=self.chat_model, contents=prompt, config=config)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(response.text or "")
        except Exception as e:  # malformed JSON from the model
            raise AIProviderError("model returned an invalid structured response") from e


# --------------------------------------------------------------------------- Fake (tests / offline)
class FakeProvider:
    """Deterministic, offline provider for tests: hashed bag-of-words vectors and a canned answer."""

    name = "fake"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls: list[dict] = []

    def embed(self, texts: list[str], *, kind: Kind) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            out.append(_normalize(v))
        return out

    def generate(self, *, system: str, prompt: str, schema: type[T], temperature: float = 0.2) -> T:
        self.calls.append({"system": system, "prompt": prompt})
        has_context = "[1]" in prompt
        return schema.model_validate({
            "language": "en",
            "answerable": has_context,
            "answer": "Answer based on the gym knowledge base." if has_context else "I don't have that information.",
            "used_sources": [1] if has_context else [],
        })


def _normalize(values) -> list[float]:
    values = list(values)
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return [x / norm for x in values]


def build_provider(settings) -> AIProvider:
    if settings.gemini_api_key is None:
        raise AIProviderError("GEMINI_API_KEY is not set (copy .env.example to .env and add your key)")
    return GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        embed_dim=settings.embed_dim,
    )
