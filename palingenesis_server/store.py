"""Durable FIFO and byte-addressable per-job output."""
import json
import os
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.root / "jobs.sqlite3", timeout=10)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def initialize(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, operation TEXT NOT NULL, args TEXT NOT NULL,
                workdir TEXT NOT NULL, processes INTEGER NOT NULL, expected_steps INTEGER,
                state TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT,
                finished_at TEXT, exit_code INTEGER, error TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state)")

    def enqueue(self, request, workdir: Path):
        job_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("""INSERT INTO jobs
                (id, operation, args, workdir, processes, expected_steps, state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)""",
                (job_id, request.operation, json.dumps(request.args), str(workdir),
                 request.processes, request.expected_steps, now()))
        return job_id

    def get(self, job_id):
        with self.connect() as db:
            return db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def recent(self, limit, offset):
        with self.connect() as db:
            return db.execute("SELECT * FROM jobs ORDER BY rowid DESC LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()

    def has_work(self):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running') LIMIT 1").fetchone()

    def claim(self):
        # Called only while holding the process-shared worker lock.
        with self.connect() as db:
            db.execute("""UPDATE jobs SET state='interrupted', finished_at=?,
                       error='Worker exited before recording the result' WHERE state='running'""", (now(),))
            job = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
            if job:
                db.execute("UPDATE jobs SET state='running', started_at=? WHERE id=?", (now(), job["id"]))
            return job

    def finish(self, job_id, code, error):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state=?, finished_at=?, exit_code=?, error=? WHERE id=?",
                       ("succeeded" if code == 0 else "failed", now(), code, error, job_id))

    def log_path(self, job_id):
        return self.root / f"{job_id}.log"

    def read_log(self, job_id, offset, limit):
        try:
            with self.log_path(job_id).open("rb") as log:
                size = os.fstat(log.fileno()).st_size
                start = max(0, size - limit) if offset is None else min(size, offset)
                log.seek(start)
                return start, log.read(min(limit, size - start)), size
        except FileNotFoundError:
            return 0, b"", 0
