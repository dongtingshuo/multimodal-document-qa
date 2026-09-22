import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .schemas import Element


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "metadata.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                  id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS elements (
                  id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                  body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS elements_document ON elements(document_id);
                CREATE TABLE IF NOT EXISTS records (
                  kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def documents(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT body FROM documents ORDER BY rowid DESC")]

    def document(self, doc_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM documents WHERE id=?", (doc_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_document(self, doc):
        with self.connect() as db:
            db.execute(
                "INSERT INTO documents VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                (doc["id"], doc["sha256"], json.dumps(doc, ensure_ascii=False)),
            )

    def replace_elements(self, doc_id: str, elements: list[Element]):
        with self.connect() as db:
            db.execute("DELETE FROM elements WHERE document_id=?", (doc_id,))
            db.executemany(
                "INSERT INTO elements VALUES (?,?,?)", [(e.id, doc_id, e.model_dump_json()) for e in elements]
            )

    def elements(self, document_ids=None):
        if document_ids == []:
            return []
        with self.connect() as db:
            sql = "SELECT e.body FROM elements e JOIN documents d ON e.document_id=d.id WHERE json_extract(d.body,'$.status')='ready'"
            params = []
            if document_ids is not None:
                sql += " AND e.document_id IN (" + ",".join("?" for _ in document_ids) + ")"
                params = document_ids
            return [Element.model_validate_json(r[0]) for r in db.execute(sql + " ORDER BY e.id", params)]

    def element(self, element_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM elements WHERE id=?", (element_id,)).fetchone()
        return Element.model_validate_json(row[0]) if row else None

    def delete(self, doc_id):
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def purge_document_history(self, doc_id):
        """Remove only conversations depending on a removed/reparsed document."""
        with self.connect() as db:
            records = [
                (r[0], r[1], json.loads(r[2]))
                for r in db.execute("SELECT kind,id,body FROM records WHERE kind IN ('sessions','answers')")
            ]
            sessions = {
                rid for kind, rid, body in records if kind == "sessions" and doc_id in body.get("scope", [])
            }
            for kind, rid, body in records:
                dependent = kind == "sessions" and rid in sessions
                if kind == "answers":
                    dependent = body.get("session_id") in sessions or any(
                        e["element"]["document_id"] == doc_id for e in body.get("evidence", [])
                    )
                if dependent:
                    db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, rid))
                    if kind == "answers":
                        db.execute("DELETE FROM records WHERE kind='retrievals' AND id=?", (rid,))

    def put(self, kind, record_id, body):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?)",
                (kind, record_id, json.dumps(body, ensure_ascii=False)),
            )

    def get(self, kind, record_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, record_id)).fetchone()
        return json.loads(row[0]) if row else None

    def records(self, kind):
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute("SELECT body FROM records WHERE kind=? ORDER BY rowid DESC", (kind,))
            ]

    def call_count(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM counters WHERE name='api_calls'").fetchone()
            return row[0] if row else 0

    def reserve_call(self, limit):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM counters WHERE name='api_calls'").fetchone()
            current = row[0] if row else 0
            if current >= limit:
                raise RuntimeError("已达到 DOCQA_MAX_API_CALLS 调用上限")
            db.execute("INSERT OR REPLACE INTO counters VALUES ('api_calls',?)", (current + 1,))
