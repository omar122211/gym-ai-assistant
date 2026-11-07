"""FastAPI service.  Run:  uvicorn app.api:app --reload"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.knowledge import KnowledgeBase, load_chunks
from app.logging_setup import setup_logging
from app.providers import AIProvider, AIProviderError, build_provider
from app.rag import Answer, RAGService

log = logging.getLogger("app.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The visitor's question, in any language")

    @field_validator("question")
    @classmethod
    def clean(cls, v: str) -> str:
        v = " ".join(v.split())  # collapse whitespace / control characters
        if not v:
            raise ValueError("question is empty")
        return v


class RateLimiter:
    """Simple in-memory sliding window per client (protects the free API quota)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def create_app(settings: Settings | None = None, provider: AIProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    state: dict = {"rag": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            prov = provider or build_provider(settings)
            kb = KnowledgeBase(settings.chroma_dir, settings.collection)
            kb.build(load_chunks(settings.knowledge_dir), prov)
            state["rag"] = RAGService(kb, prov, settings.top_k, settings.min_similarity)
            log.info("ready", extra={"fields": {"provider": prov.name, "chunks": kb.count()}})
        except AIProviderError as e:
            state["error"] = str(e)
            log.error("AI provider not available", extra={"fields": {"reason": str(e)}})
        yield

    app = FastAPI(title="Gym AI Assistant", version="0.1.0", lifespan=lifespan)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        log.info("request", extra={"fields": {
            "id": rid, "method": request.method, "path": request.url.path,
            "status": response.status_code, "ms": int((time.perf_counter() - started) * 1000),
        }})
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/health")
    def health():
        rag: RAGService | None = state["rag"]
        return {
            "status": "ok" if rag else "degraded",
            "provider": rag.provider.name if rag else None,
            "chunks": rag.kb.count() if rag else 0,
            "error": state["error"],
        }

    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request):
        if len(body.question) > settings.max_question_chars:
            raise HTTPException(422, f"question is longer than {settings.max_question_chars} characters")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "Too many questions — please wait a minute.")
        rag: RAGService | None = state["rag"]
        if rag is None:
            raise HTTPException(503, "The assistant is not configured yet.")
        try:
            return rag.answer(body.question)
        except AIProviderError:
            log.exception("AI provider failed")
            raise HTTPException(503, "The AI service is busy right now — please try again shortly.")

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unexpected error")
        return JSONResponse(status_code=500, content={"detail": "Something went wrong."})

    return app


app = create_app()
