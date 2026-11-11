"""FastAPI service.  Run:  uvicorn app.api:app --reload"""
from __future__ import annotations

import logging
import secrets
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field, field_validator

from app.agent import ChatResult, GymAgent
from app.config import Settings, get_settings
from app.gym_db import GymDB
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


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The member's message, in any language")
    # Random id from the client (or created by the server). It is the only key to the memory, so keep it unguessable.
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{16,64}$")

    @field_validator("message")
    @classmethod
    def clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("message is empty")
        return v


class TicketUpdate(BaseModel):
    status: Literal["open", "in_progress", "resolved"]


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
    state: dict = {"rag": None, "agent": None, "error": None}
    db = GymDB(settings.gym_db_path)  # works without an AI key, so /schedule is always available

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        memory_conn = None
        try:
            prov = provider or build_provider(settings)
            kb = KnowledgeBase(settings.chroma_dir, settings.collection)
            kb.build(load_chunks(settings.knowledge_dir), prov)
            state["rag"] = RAGService(kb, prov, settings.top_k, settings.min_similarity)
            settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            memory_conn = sqlite3.connect(str(settings.memory_db_path), check_same_thread=False)
            state["agent"] = GymAgent(state["rag"], db, prov, SqliteSaver(memory_conn))
            log.info("ready", extra={"fields": {"provider": prov.name, "chunks": kb.count()}})
        except AIProviderError as e:
            state["error"] = str(e)
            log.error("AI provider not available", extra={"fields": {"reason": str(e)}})
        yield
        if memory_conn is not None:
            memory_conn.close()

    app = FastAPI(title="Gym AI Assistant", version="0.2.0", lifespan=lifespan)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    def guard(text: str, request: Request) -> None:
        if len(text) > settings.max_question_chars:
            raise HTTPException(422, f"message is longer than {settings.max_question_chars} characters")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "Too many messages — please wait a minute.")

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
        """One-shot question answered from the knowledge base (no memory, no tools)."""
        guard(body.question, request)
        rag: RAGService | None = state["rag"]
        if rag is None:
            raise HTTPException(503, "The assistant is not configured yet.")
        try:
            return rag.answer(body.question)
        except AIProviderError:
            log.exception("AI provider failed")
            raise HTTPException(503, "The AI service is busy right now — please try again shortly.")

    @app.post("/chat", response_model=ChatResult)
    def chat(body: ChatRequest, request: Request):
        """Conversation with the agent: questions, class schedule, bookings and complaints, with memory."""
        guard(body.message, request)
        agent: GymAgent | None = state["agent"]
        if agent is None:
            raise HTTPException(503, "The assistant is not configured yet.")
        try:
            return agent.chat(body.message, body.conversation_id or uuid.uuid4().hex)
        except AIProviderError:
            log.exception("AI provider failed")
            raise HTTPException(503, "The AI service is busy right now — please try again shortly.")

    @app.get("/schedule")
    def schedule(class_name: str | None = Query(default=None, max_length=40),
                 day: str | None = Query(default=None, max_length=12)):
        """Class timetable (demo data) with free spots."""
        return {"demo": True, "sessions": db.schedule(class_name, day)}

    # ---------------------------------------------------------------- staff tools
    failed_logins = RateLimiter(5)  # wrong tokens per client per minute (stops guessing)

    def require_staff(request: Request, x_staff_token: str | None = Header(default=None)) -> None:
        """Every /staff route needs the STAFF_TOKEN in the X-Staff-Token header."""
        expected = settings.staff_token.get_secret_value() if settings.staff_token else ""
        if not expected:
            raise HTTPException(503, "Staff tools are turned off (set STAFF_TOKEN).")
        if not x_staff_token or not secrets.compare_digest(x_staff_token.encode(), expected.encode()):
            client = request.client.host if request.client else "unknown"
            if not failed_logins.allow(client):
                raise HTTPException(429, "Too many wrong tokens — wait a minute.")
            raise HTTPException(401, "Wrong or missing staff token.")

    staff = [Depends(require_staff)]

    @app.get("/staff/report", dependencies=staff)
    def staff_report():
        """Totals, bookings per class, busiest days and tickets by status."""
        return db.report()

    @app.get("/staff/bookings", dependencies=staff)
    def staff_bookings(session_id: int | None = Query(default=None, ge=1)):
        return {"bookings": db.list_bookings(session_id)}

    @app.delete("/staff/bookings/{booking_id}", dependencies=staff)
    def staff_cancel_booking(booking_id: int):
        if not db.cancel_booking(booking_id):
            raise HTTPException(404, "No booking with that id.")
        log.info("booking cancelled", extra={"fields": {"booking_id": booking_id}})
        return {"cancelled": booking_id}

    @app.get("/staff/tickets", dependencies=staff)
    def staff_tickets(status: Literal["open", "in_progress", "resolved"] | None = None):
        return {"tickets": db.list_tickets(status)}

    @app.patch("/staff/tickets/{ticket_id}", dependencies=staff)
    def staff_update_ticket(ticket_id: int, body: TicketUpdate):
        if not db.update_ticket_status(ticket_id, body.status):
            raise HTTPException(404, "No ticket with that id.")
        log.info("ticket updated", extra={"fields": {"ticket_id": ticket_id, "status": body.status}})
        return {"id": ticket_id, "status": body.status}

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unexpected error")
        return JSONResponse(status_code=500, content={"detail": "Something went wrong."})

    return app


app = create_app()
