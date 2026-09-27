"""
SQLite store for the officer workflow (F28) and farmer dissemination simulator (F27).

Tables
------
reviews      (issue_date, gp_code) -> status draft|approved|rejected, officer, note, edited advisory text
audit_log    append-only record of every officer / system action (who, what, when)
subscribers  mock farmer registrations: GP, language, channel. Phone numbers are stored
             only as a salted hash plus the last 2 digits; no real messages are ever sent.
outbox       simulated SMS / WhatsApp / IVR messages generated for approved bulletins
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  issue_date TEXT NOT NULL, gp_code TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
  officer TEXT, note TEXT, edits TEXT, updated_utc TEXT,
  PRIMARY KEY (issue_date, gp_code));
CREATE TABLE IF NOT EXISTS issue_status (
  issue_date TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'draft', officer TEXT, published_utc TEXT);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL, actor TEXT, action TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS subscribers (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone_hash TEXT NOT NULL, phone_last2 TEXT,
  gp_code TEXT NOT NULL, lang TEXT NOT NULL, channel TEXT NOT NULL, created_utc TEXT,
  UNIQUE (phone_hash, gp_code, channel));
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL, issue_date TEXT, subscriber_id INTEGER,
  gp_code TEXT, channel TEXT, lang TEXT, message TEXT, chars INTEGER, segments INTEGER,
  status TEXT NOT NULL DEFAULT 'simulated');
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---- audit ---------------------------------------------------------------------------------
    def log(self, actor: str | None, action: str, detail: dict | None = None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO audit_log (ts_utc, actor, action, detail) VALUES (?,?,?,?)",
                      (now(), actor, action, json.dumps(detail or {}, ensure_ascii=False)))

    def audit(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]

    # ---- reviews -------------------------------------------------------------------------------
    def review(self, issue: str, gp: str, status: str, officer: str, note: str = "", edits: dict | None = None) -> dict:
        if status not in ("draft", "approved", "rejected"):
            raise ValueError("status must be draft|approved|rejected")
        with self.conn() as c:
            c.execute("""INSERT INTO reviews (issue_date, gp_code, status, officer, note, edits, updated_utc)
                         VALUES (?,?,?,?,?,?,?) ON CONFLICT(issue_date, gp_code) DO UPDATE SET
                         status=excluded.status, officer=excluded.officer, note=excluded.note,
                         edits=excluded.edits, updated_utc=excluded.updated_utc""",
                      (issue, gp, status, officer, note, json.dumps(edits or {}, ensure_ascii=False), now()))
        self.log(officer, f"review:{status}", {"issue_date": issue, "gp_code": gp, "note": note, "edits": edits})
        return self.get_review(issue, gp)

    def get_review(self, issue: str, gp: str) -> dict:
        with self.conn() as c:
            r = c.execute("SELECT * FROM reviews WHERE issue_date=? AND gp_code=?", (issue, gp)).fetchone()
        if not r:
            return {"issue_date": issue, "gp_code": gp, "status": "auto"}
        d = dict(r)
        d["edits"] = json.loads(d["edits"] or "{}")
        return d

    def reviews(self, issue: str) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM reviews WHERE issue_date=?", (issue,))]

    def publish(self, issue: str, officer: str) -> dict:
        with self.conn() as c:
            c.execute("""INSERT INTO issue_status (issue_date, status, officer, published_utc) VALUES (?,?,?,?)
                         ON CONFLICT(issue_date) DO UPDATE SET status=excluded.status, officer=excluded.officer,
                         published_utc=excluded.published_utc""", (issue, "published", officer, now()))
        self.log(officer, "publish", {"issue_date": issue})
        return self.issue_status(issue)

    def issue_status(self, issue: str) -> dict:
        with self.conn() as c:
            r = c.execute("SELECT * FROM issue_status WHERE issue_date=?", (issue,)).fetchone()
        return dict(r) if r else {"issue_date": issue, "status": "draft"}

    # ---- subscribers / outbox ------------------------------------------------------------------
    @staticmethod
    def _hash(phone: str) -> str:
        salt = os.environ.get("AGROMET_PHONE_SALT", "sih074-demo-salt")
        return hashlib.sha256((salt + phone).encode()).hexdigest()

    def subscribe(self, name: str, phone: str, gp: str, lang: str, channel: str) -> dict:
        digits = "".join(ch for ch in phone if ch.isdigit())
        if len(digits) < 10:
            raise ValueError("phone must have at least 10 digits")
        if channel not in ("sms", "whatsapp", "ivr"):
            raise ValueError("channel must be sms|whatsapp|ivr")
        with self.conn() as c:
            c.execute("""INSERT OR IGNORE INTO subscribers (name, phone_hash, phone_last2, gp_code, lang, channel,
                         created_utc) VALUES (?,?,?,?,?,?,?)""",
                      (name, self._hash(digits), digits[-2:], gp, lang, channel, now()))
            r = c.execute("SELECT id, name, phone_last2, gp_code, lang, channel, created_utc FROM subscribers "
                          "WHERE phone_hash=? AND gp_code=? AND channel=?", (self._hash(digits), gp, channel)).fetchone()
        self.log("farmer", "subscribe", {"gp_code": gp, "lang": lang, "channel": channel})
        return dict(r)

    def subscribers(self, gp: str | None = None) -> list[dict]:
        q = "SELECT id, name, phone_last2, gp_code, lang, channel, created_utc FROM subscribers"
        with self.conn() as c:
            rows = c.execute(q + (" WHERE gp_code=?" if gp else ""), ((gp,) if gp else ()))
            return [dict(r) for r in rows]

    def queue(self, issue: str, sub: dict, message: str) -> None:
        segs = 1 if len(message) <= 70 else -(-len(message) // 67)  # Unicode SMS: 70 / 67 per part
        with self.conn() as c:
            c.execute("""INSERT INTO outbox (ts_utc, issue_date, subscriber_id, gp_code, channel, lang, message, chars,
                         segments) VALUES (?,?,?,?,?,?,?,?,?)""",
                      (now(), issue, sub["id"], sub["gp_code"], sub["channel"], sub["lang"], message, len(message), segs))

    def outbox(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM outbox ORDER BY id DESC LIMIT ?", (limit,))]
