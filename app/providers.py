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

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class AIProvider(Protocol):
    name: str

    def embed(self, texts: list[str], *, kind: Kind) -> list[list[float]]: ...

    def generate(self, *, system: str, prompt: str, schema: type[T], temperature: float = 0.2) -> T: ...


# --------------------------------------------------------------------------- Gemini
class GeminiProvider:
    """Google Gemini API (google-genai SDK)."""

    name = "gemini"
    _RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, chat_model: str, embed_model: str, embed_dim: int,
                 fallback_chat_model: str | None = None, max_retries: int = 2):
        from google import genai  # imported here so tests don't need the SDK configured

        self._genai_types = __import__("google.genai.types", fromlist=["types"])
        self._errors = __import__("google.genai.errors", fromlist=["errors"])
        self.client = genai.Client(api_key=api_key)
        self.chat_model, self.embed_model, self.embed_dim = chat_model, embed_model, embed_dim
        self.fallback_chat_model = fallback_chat_model
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
                raise AIProviderError(f"Gemini API error {code}", code) from e
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
        models = [self.chat_model] + ([self.fallback_chat_model] if self.fallback_chat_model else [])
        for i, model in enumerate(models):
            try:
                response = self._call(self.client.models.generate_content, model=model, contents=prompt, config=config)
                break
            except AIProviderError as e:
                # model overloaded or out of quota → try the next model instead of failing the user
                if e.code in (429, 503) and i + 1 < len(models):
                    log.warning("gemini fallback", extra={"fields": {"from": model, "to": models[i + 1], "code": e.code}})
                    continue
                raise
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
        self.calls.append({"system": system, "prompt": prompt, "schema": schema.__name__})
        fields = schema.model_fields
        if "intent" in fields:  # agent router
            return schema.model_validate(self._route(system, prompt))
        if "reply" in fields:  # agent reply: echo the tool result so tests can check it
            result = re.search(r"<result>\n(.*?)\n</result>", prompt, re.S)
            return schema.model_validate({"reply": f"[fake] {result.group(1) if result else ''}"})
        has_context = "[1]" in prompt
        return schema.model_validate({
            "language": "en",
            "answerable": has_context,
            "answer": "Answer based on the gym knowledge base." if has_context else "I don't have that information.",
            "used_sources": [1] if has_context else [],
        })

    # A tiny keyword router — good enough to test the graph without a real model.
    _INTENTS = [("booking", ("book", "reserve", "sign me up")), ("schedule", ("schedule", "when is", "timetable")),
                ("complaint", ("complain", "broken", "dirty", "rude")), ("greeting", ("hello", "hi ", "thanks")),
                ("out_of_scope", ("weather", "bitcoin", "football"))]

    def _route(self, system: str, prompt: str) -> dict:
        latest = re.search(r"<latest>\n(.*?)\n</latest>", prompt, re.S).group(1)
        convo = re.search(r"<conversation>\n(.*?)\n</conversation>", prompt, re.S).group(1)
        user_lines = [line[6:] for line in convo.splitlines() if line.startswith("User: ")]
        classes = re.search(r"Known classes: (.*?)\. Days:", system).group(1).split(", ")
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        def intent_of(text: str) -> str | None:
            low = f" {text.lower()} "
            return next((name for name, words in self._INTENTS if any(w in low for w in words)), None)

        intent = intent_of(latest)
        if intent is None:  # a follow-up like "Friday" or "my name is ..." continues the last request
            previous = next((i for i in map(intent_of, reversed(user_lines)) if i), None)
            intent = "booking" if previous == "booking" else None
        text = " ".join(user_lines + [latest])
        low = text.lower()
        # latest-mentioned class wins; on the same end position the longer name wins ("kick-boxing" over "boxing")
        found_class = max(((low.rfind(c.lower()) + len(c), len(c), c) for c in classes if c.lower() in low),
                          default=(-1, 0, None))[2]
        found_day = max(((low.rfind(d), d.title()) for d in days if d in low), default=(-1, None))[1]
        name = re.findall(r"my name is (\w+)", text, re.I)
        return {"intent": intent or "question", "language": "en", "standalone_question": latest,
                "class_name": found_class, "day": found_day, "member_name": name[-1] if name else None,
                "complaint_summary": latest if intent == "complaint" else None}


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
        fallback_chat_model=settings.fallback_chat_model,
    )
