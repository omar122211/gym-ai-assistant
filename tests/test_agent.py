"""Offline tests for the agent and its tools (FakeProvider: no key, no network)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import GymAgent
from app.api import create_app
from app.config import ROOT, Settings
from app.gym_db import CAPACITY, GymDB
from app.knowledge import KnowledgeBase, load_chunks
from app.providers import FakeProvider
from app.rag import RAGService


@pytest.fixture
def db() -> GymDB:
    return GymDB(":memory:")


@pytest.fixture
def agent(tmp_path: Path, db: GymDB) -> GymAgent:
    provider = FakeProvider()
    kb = KnowledgeBase(tmp_path / "chroma", "test_kb")
    kb.build(load_chunks(ROOT / "knowledge"), provider)
    return GymAgent(RAGService(kb, provider, min_similarity=0.0), db, provider, InMemorySaver())


def action_in_reply(reply: str) -> dict:
    """FakeProvider echoes the tool result as '[fake] {json}'."""
    return json.loads(reply.removeprefix("[fake] "))


# ---------------------------------------------------------------- database tools
def test_schedule_filters_and_exact_class_match(db: GymDB):
    boxing = db.schedule("boxing")
    assert {s["class_name"] for s in boxing} == {"Boxing"}, "'boxing' must not include kick-boxing"
    assert [s["day"] for s in db.schedule(day="Monday")] == ["Monday", "Monday"]
    assert all(s["spots_left"] == CAPACITY for s in db.schedule())


def test_booking_rules(db: GymDB):
    session = db.schedule("Yoga", "Monday")[0]
    first = db.book(session["id"], "Omar")
    assert first["status"] == "booked" and first["spots_left"] == CAPACITY - 1
    assert db.book(session["id"], "  omar ")["status"] == "already_booked"
    assert db.book(9999, "Omar")["status"] == "not_found"


def test_full_class(db: GymDB):
    session = db.schedule("Pilates", "Tuesday")[0]
    for i in range(CAPACITY):
        db.book(session["id"], f"member {i}")
    assert db.book(session["id"], "late member")["status"] == "full"


def test_user_text_is_never_sql(db: GymDB):
    db.book(db.schedule("Yoga")[0]["id"], "x'); DROP TABLE bookings; --")
    assert db.counts()["bookings"] == 1


# ---------------------------------------------------------------- agent routing
def test_question_goes_to_rag(agent: GymAgent):
    r = agent.chat("How much is the yearly membership?", "conv-question-0001")
    assert r.intent == "question" and r.grounded and r.sources


def test_schedule_intent(agent: GymAgent):
    r = agent.chat("What is the schedule for calisthenics?", "conv-schedule-0001")
    action = action_in_reply(r.reply)
    assert r.intent == "schedule" and action["type"] == "schedule"
    assert {s["class_name"] for s in action["sessions"]} == {"Calisthenics"}


def test_out_of_scope_and_greeting(agent: GymAgent):
    assert agent.chat("What's the weather in Istanbul?", "conv-scope-00001").action == {"type": "out_of_scope"}
    assert agent.chat("hello there", "conv-scope-00002").action == {"type": "greeting"}


def test_complaint_creates_ticket(agent: GymAgent, db: GymDB):
    r = agent.chat("The shower in the changing room is broken", "conv-complaint-01")
    assert r.action["type"] == "ticket" and db.counts()["tickets"] == 1


# ---------------------------------------------------------------- multi-turn booking + memory
def test_booking_asks_for_missing_details_and_remembers(agent: GymAgent, db: GymDB):
    cid = "conv-booking-00001"
    step1 = agent.chat("I want to book a boxing class", cid)
    assert step1.action["type"] == "need_info" and step1.action["missing"] == ["day"]

    step2 = agent.chat("Friday please", cid)  # class remembered from turn 1
    assert step2.action["type"] == "need_info" and step2.action["missing"] == ["name for the booking"]

    step3 = agent.chat("my name is Omar", cid)  # class + day remembered
    assert step3.action["type"] == "booked"
    assert step3.action["session"]["class_name"] == "Boxing" and step3.action["session"]["day"] == "Friday"
    assert db.counts()["bookings"] == 1


def test_conversations_are_isolated(agent: GymAgent):
    agent.chat("I want to book a boxing class", "conv-isolated-0001")
    other = agent.chat("Friday please", "conv-isolated-0002")  # new conversation: no boxing context
    assert other.intent == "question"


def test_unknown_day_for_class(agent: GymAgent):
    r = agent.chat("book yoga on Sunday, my name is Omar", "conv-unknownday-1")
    assert r.action["type"] == "not_found" and r.action["other_sessions"]


# ---------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(gemini_api_key=None, chroma_dir=tmp_path / "chroma", gym_db_path=tmp_path / "gym.db",
                        memory_db_path=tmp_path / "memory.db", min_similarity=0.0)
    with TestClient(create_app(settings, FakeProvider())) as c:
        yield c


def test_chat_endpoint_keeps_memory_across_requests(client: TestClient):
    first = client.post("/chat", json={"message": "I want to book a boxing class"}).json()
    cid = first["conversation_id"]
    second = client.post("/chat", json={"message": "Friday please, my name is Omar", "conversation_id": cid}).json()
    assert second["action"]["type"] == "booked"
    assert client.get("/schedule", params={"class_name": "Boxing", "day": "Friday"}).json()["sessions"][0]["spots_left"] == CAPACITY - 1


def test_chat_validation(client: TestClient):
    assert client.post("/chat", json={"message": "  "}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "conversation_id": "../../etc"}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 501}).status_code == 422


def test_schedule_works_without_api_key(tmp_path: Path):
    settings = Settings(gemini_api_key=None, chroma_dir=tmp_path / "c", gym_db_path=tmp_path / "g.db",
                        memory_db_path=tmp_path / "m.db")
    with TestClient(create_app(settings)) as c:
        assert c.get("/schedule").json()["sessions"]
        assert c.post("/chat", json={"message": "hello"}).status_code == 503
