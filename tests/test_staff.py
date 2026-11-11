"""Schema migrations, staff CRUD on the gym database, and the protected /staff API."""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.gym_db import LATEST_VERSION, SCHEMA_V1, GymDB
from app.providers import FakeProvider

TOKEN = "test-staff-token-1234567890"


@pytest.fixture
def db(tmp_path: Path) -> GymDB:
    return GymDB(tmp_path / "gym.db")


# ---------------------------------------------------------------- migrations
def test_new_database_is_on_the_latest_version(db: GymDB):
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == LATEST_VERSION
    indexes = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"idx_bookings_session", "idx_tickets_status"} <= indexes
    assert "updated_at" in {r[1] for r in db.conn.execute("PRAGMA table_info(tickets)")}


def test_old_database_is_upgraded_without_losing_data(tmp_path: Path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)  # a database made by the previous version of the app (no user_version)
    old.executescript(SCHEMA_V1)
    old.execute("INSERT INTO tickets (summary, created_at) VALUES ('Broken shower', '2026-01-01T10:00:00+00:00')")
    old.commit()
    old.close()

    db = GymDB(path)
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == LATEST_VERSION
    [ticket] = db.list_tickets()
    assert ticket["summary"] == "Broken shower" and ticket["status"] == "open" and ticket["updated_at"] is None


def test_migrations_run_only_once(tmp_path: Path):
    path = tmp_path / "gym.db"
    GymDB(path).create_ticket("first", "en", None)
    db = GymDB(path)  # opening again must not re-run migrations or re-seed the schedule
    assert db.counts()["tickets"] == 1
    assert db.counts()["sessions"] == 14


# ---------------------------------------------------------------- CRUD
def test_list_and_cancel_bookings(db: GymDB):
    sid = db.schedule("Yoga")[0]["id"]
    booking = db.book(sid, "Ayşe")
    [row] = db.list_bookings(sid)
    assert row["member_name"] == "Ayşe" and row["class_name"] == "Yoga"
    assert db.cancel_booking(booking["booking_id"]) is True
    assert db.list_bookings(sid) == []
    assert db.cancel_booking(booking["booking_id"]) is False  # already gone
    assert db.schedule("Yoga")[0]["spots_left"] == 12          # the seat is free again


def test_ticket_status_update(db: GymDB):
    tid = db.create_ticket("AC is too cold", "en", None)
    assert db.update_ticket_status(tid, "in_progress") is True
    [t] = db.list_tickets("in_progress")
    assert t["id"] == tid and t["updated_at"]
    assert db.list_tickets("open") == []
    assert db.update_ticket_status(9999, "resolved") is False
    with pytest.raises(ValueError):
        db.update_ticket_status(tid, "deleted")


def test_report_aggregates(db: GymDB):
    boxing = db.schedule("Boxing")
    db.book(boxing[0]["id"], "A")
    db.book(boxing[1]["id"], "B")
    db.book(db.schedule("Yoga")[0]["id"], "C")
    db.create_ticket("x", "en", None)
    r = db.report()
    assert r["per_class"][0] == {"class_name": "Boxing", "bookings": 2, "seats": 24}
    assert sum(d["bookings"] for d in r["busiest_days"]) == 3
    assert r["tickets_by_status"] == {"open": 1, "in_progress": 0, "resolved": 0}
    assert r["totals"]["bookings"] == 3


def test_foreign_keys_are_enforced(db: GymDB):
    with pytest.raises(sqlite3.IntegrityError):
        with db.conn:
            db.conn.execute("INSERT INTO bookings (session_id, member_name, created_at) VALUES (999, 'x', 'now')")


# ---------------------------------------------------------------- staff API
def make_client(tmp_path: Path, token: str | None) -> TestClient:
    s = Settings(gemini_api_key=None, staff_token=token, chroma_dir=tmp_path / "chroma",
                 gym_db_path=tmp_path / "gym.db", memory_db_path=tmp_path / "memory.db")
    return TestClient(create_app(s, provider=FakeProvider()))


def test_staff_routes_are_off_without_a_token(tmp_path: Path):
    with make_client(tmp_path, None) as c:
        assert c.get("/staff/report").status_code == 503


def test_staff_routes_need_the_right_token(tmp_path: Path):
    with make_client(tmp_path, TOKEN) as c:
        assert c.get("/staff/report").status_code == 401
        assert c.get("/staff/report", headers={"X-Staff-Token": "wrong"}).status_code == 401
        assert c.get("/staff/report", headers={"X-Staff-Token": TOKEN}).status_code == 200


def test_wrong_tokens_are_rate_limited(tmp_path: Path):
    with make_client(tmp_path, TOKEN) as c:
        codes = [c.get("/staff/tickets", headers={"X-Staff-Token": f"guess{i}"}).status_code for i in range(7)]
        assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]


def test_staff_crud_over_http(tmp_path: Path):
    h = {"X-Staff-Token": TOKEN}
    with make_client(tmp_path, TOKEN) as c:
        db = GymDB(tmp_path / "gym.db")  # same file the app uses
        booking = db.book(db.schedule("Pilates")[0]["id"], "Mert")
        tid = db.create_ticket("Locker 12 is broken", "en", None)

        assert len(c.get("/staff/bookings", headers=h).json()["bookings"]) == 1
        assert c.delete(f"/staff/bookings/{booking['booking_id']}", headers=h).json() == {"cancelled": booking["booking_id"]}
        assert c.delete(f"/staff/bookings/{booking['booking_id']}", headers=h).status_code == 404

        assert c.patch(f"/staff/tickets/{tid}", json={"status": "resolved"}, headers=h).status_code == 200
        assert c.patch(f"/staff/tickets/{tid}", json={"status": "deleted"}, headers=h).status_code == 422
        assert c.get("/staff/tickets?status=resolved", headers=h).json()["tickets"][0]["id"] == tid
        assert c.get("/staff/report", headers=h).json()["tickets_by_status"]["resolved"] == 1
