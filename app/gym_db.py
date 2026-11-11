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

TICKET_STATUSES = ("open", "in_progress", "resolved")

# ---------------------------------------------------------------- schema management
# Versioned migrations. SQLite keeps the applied version in `PRAGMA user_version`; on start-up every migration
# newer than that runs once, in order, inside a transaction. Never edit an old migration: add a new one.
SCHEMA_V1 = """
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


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _v1_initial_schema(conn: sqlite3.Connection) -> None:
    for statement in filter(str.strip, SCHEMA_V1.split(";")):
        conn.execute(statement)


def _v2_indexes_and_ticket_updates(conn: sqlite3.Connection) -> None:
    # Indexes for the lookups the app does most: bookings by session, tickets by status
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bookings_session ON bookings(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status)")
    if "updated_at" not in _columns(conn, "tickets"):  # staff can now change a ticket's status
        conn.execute("ALTER TABLE tickets ADD COLUMN updated_at TEXT")


MIGRATIONS = [  # (version, function) — append only
    (1, _v1_initial_schema),
    (2, _v2_indexes_and_ticket_updates),
]
LATEST_VERSION = MIGRATIONS[-1][0]


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database up to LATEST_VERSION. Returns the version it started from."""
    start = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, apply in MIGRATIONS:
        if version > start:
            # One explicit transaction per migration: it fully applies or not at all.
            # (Python's sqlite3 doesn't open a transaction for CREATE/ALTER by itself.)
            conn.execute("BEGIN")
            try:
                apply(conn)
                conn.execute(f"PRAGMA user_version = {int(version)}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    return start


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
        self.conn.execute("PRAGMA foreign_keys = ON")  # SQLite only enforces REFERENCES when this is on
        self.lock = threading.Lock()
        with self.lock:
            migrate(self.conn)
        with self.lock, self.conn:
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

    # ------------------------------------------------------------------ staff tools (read / update / delete)
    def list_bookings(self, session_id: int | None = None) -> list[dict]:
        sql = """SELECT b.id, b.member_name, b.created_at, s.id AS session_id, s.class_name, s.day, s.time
                 FROM bookings b JOIN sessions s ON s.id = b.session_id"""
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE b.session_id = ?"
            params = (session_id,)
        sql += " ORDER BY s.id, b.created_at"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def cancel_booking(self, booking_id: int) -> bool:
        """DELETE one booking. Returns False if there was no such booking."""
        with self.lock, self.conn:
            return self.conn.execute("DELETE FROM bookings WHERE id = ?", (booking_id,)).rowcount == 1

    def list_tickets(self, status: str | None = None) -> list[dict]:
        if status is not None and status not in TICKET_STATUSES:
            raise ValueError(f"status must be one of {TICKET_STATUSES}")
        sql = "SELECT id, summary, language, status, created_at, updated_at FROM tickets"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY created_at DESC, id DESC"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def update_ticket_status(self, ticket_id: int, status: str) -> bool:
        """UPDATE a ticket's status. Returns False if there was no such ticket."""
        if status not in TICKET_STATUSES:
            raise ValueError(f"status must be one of {TICKET_STATUSES}")
        with self.lock, self.conn:
            cur = self.conn.execute("UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), ticket_id))
            return cur.rowcount == 1

    def report(self) -> dict:
        """Aggregates for the staff page (GROUP BY / HAVING / ORDER BY)."""
        per_class = self.conn.execute(
            """SELECT s.class_name,
                      COUNT(b.id) AS bookings,
                      (SELECT SUM(capacity) FROM sessions x WHERE x.class_name = s.class_name) AS seats
               FROM sessions s LEFT JOIN bookings b ON b.session_id = s.id
               GROUP BY s.class_name
               ORDER BY bookings DESC, s.class_name""").fetchall()
        busiest_days = self.conn.execute(
            """SELECT s.day, COUNT(*) AS bookings
               FROM bookings b JOIN sessions s ON s.id = b.session_id
               GROUP BY s.day HAVING COUNT(*) > 0
               ORDER BY bookings DESC""").fetchall()
        tickets = self.conn.execute("SELECT status, COUNT(*) AS n FROM tickets GROUP BY status").fetchall()
        by_status = {s: 0 for s in TICKET_STATUSES} | {r["status"]: r["n"] for r in tickets}
        return {
            "totals": self.counts(),
            "per_class": [dict(r) for r in per_class],
            "busiest_days": [dict(r) for r in busiest_days],
            "tickets_by_status": by_status,
        }

    def counts(self) -> dict:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("sessions", "bookings", "tickets")}
