"""Gym database (SQLite): class schedule, bookings and support tickets — the agent's tools.

The schedule below is DEMO data: the Fitness Abbas website has no public timetable.
All queries use placeholders (?) — user text never becomes SQL.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# (class name, trainer, day, time) — trainers and classes come from the gym's website; days/times are demo data
DEMO_SCHEDULE = [
    ("Yoga", None, "Monday", "18:00"), ("Yoga", None, "Thursday", "18:00"),
    ("Pilates", None, "Tuesday", "18:00"), ("Pilates", None, "Saturday", "10:00"),
    ("Cycling", None, "Wednesday", "19:00"), ("Cycling", None, "Sunday", "10:00"),
    ("Boxing", "Mike Tayson", "Monday", "20:00"), ("Boxing", "Mike Tayson", "Friday", "20:00"),
    ("Kick-boxing", "Mohammad Ali Klay", "Tuesday", "20:00"), ("Kick-boxing", "Mohammad Ali Klay", "Saturday", "12:00"),
    ("Calisthenics", "Ezz", "Wednesday", "17:00"), ("Calisthenics", "Ezz", "Sunday", "12:00"),
    ("Weight lifting", "Adnan", "Thursday", "20:00"), ("Weight lifting", "Adnan", "Saturday", "16:00"),
]
CAPACITY = 12

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  class_name TEXT NOT NULL, trainer TEXT, day TEXT NOT NULL, time TEXT NOT NULL,
  capacity INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS bookings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  member_name TEXT NOT NULL, conversation_id TEXT, created_at TEXT NOT NULL,
  UNIQUE (session_id, member_name)
);
CREATE TABLE IF NOT EXISTS tickets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  summary TEXT NOT NULL, language TEXT, conversation_id TEXT, status TEXT NOT NULL DEFAULT 'open',
  created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: str | None) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


class GymDB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.executescript(SCHEMA)
            if self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0:
                self.conn.executemany(
                    "INSERT INTO sessions (class_name, trainer, day, time, capacity) VALUES (?, ?, ?, ?, ?)",
                    [(c, t, d, h, CAPACITY) for c, t, d, h in DEMO_SCHEDULE],
                )

    # ------------------------------------------------------------------ schedule
    def class_names(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT class_name FROM sessions ORDER BY class_name")]

    def schedule(self, class_name: str | None = None, day: str | None = None) -> list[dict]:
        rows = self.conn.execute(
            """SELECT s.id, s.class_name, s.trainer, s.day, s.time, s.capacity,
                      s.capacity - COUNT(b.id) AS spots_left
               FROM sessions s LEFT JOIN bookings b ON b.session_id = s.id
               GROUP BY s.id ORDER BY s.id"""
        ).fetchall()
        out = [dict(r) for r in rows]
        if class_name:
            wanted = _norm(class_name)
            exact = [r for r in out if _norm(r["class_name"]) == wanted]  # "boxing" must not also match "kick-boxing"
            out = exact or [r for r in out if wanted and (wanted in _norm(r["class_name"]) or _norm(r["class_name"]) in wanted)]
        if day:
            out = [r for r in out if r["day"].lower() == day.lower()]
        return sorted(out, key=lambda r: (DAYS.index(r["day"]), r["time"]))

    # ------------------------------------------------------------------ actions
    def book(self, session_id: int, member_name: str, conversation_id: str | None = None) -> dict:
        member_name = " ".join(member_name.split())[:80]
        with self.lock, self.conn:
            row = self.conn.execute(
                """SELECT s.*, s.capacity - (SELECT COUNT(*) FROM bookings WHERE session_id = s.id) AS spots_left
                   FROM sessions s WHERE s.id = ?""", (session_id,)).fetchone()
            if row is None:
                return {"status": "not_found"}
            session = {k: row[k] for k in ("id", "class_name", "trainer", "day", "time")}
            if self.conn.execute("SELECT 1 FROM bookings WHERE session_id = ? AND lower(member_name) = lower(?)",
                                 (session_id, member_name)).fetchone():
                return {"status": "already_booked", "session": session}
            if row["spots_left"] <= 0:
                return {"status": "full", "session": session}
            cur = self.conn.execute(
                "INSERT INTO bookings (session_id, member_name, conversation_id, created_at) VALUES (?, ?, ?, ?)",
                (session_id, member_name, conversation_id, _now()))
            return {"status": "booked", "booking_id": cur.lastrowid, "session": session, "spots_left": row["spots_left"] - 1}

    def create_ticket(self, summary: str, language: str | None, conversation_id: str | None) -> int:
        with self.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO tickets (summary, language, conversation_id, created_at) VALUES (?, ?, ?, ?)",
                (summary[:1000], language, conversation_id, _now()))
            return cur.lastrowid

    def counts(self) -> dict:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("sessions", "bookings", "tickets")}
