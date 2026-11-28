"""Local component generation jobs and their durable review state."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


class ComponentJobs:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, component TEXT NOT NULL, parameters TEXT NOT NULL,
                status TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
                owner INTEGER, owner_started REAL, error TEXT, result TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                job TEXT NOT NULL, number INTEGER NOT NULL, parameters TEXT NOT NULL,
                status TEXT NOT NULL, error TEXT, result TEXT, PRIMARY KEY (job, number)
            )""")

    def history(self, identifier):
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM attempts WHERE job = ? ORDER BY number", (identifier,))
            return [{**dict(row), "parameters": json.loads(row["parameters"])} for row in rows]

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def add(self, component, parameters):
        identifier = uuid4().hex
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO jobs (id, component, parameters, status) VALUES (?, ?, ?, 'pending')",
                       (identifier, component, json.dumps(parameters, ensure_ascii=False)))
        return identifier

    def entries(self):
        import psutil

        with closing(self._connect()) as db, db:
            for row in db.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall():
                try:
                    alive = psutil.Process(row["owner"]).create_time() == row["owner_started"]
                except psutil.NoSuchProcess:
                    alive = False
                except psutil.AccessDenied:
                    continue
                if not alive:
                    changed = db.execute("""UPDATE jobs SET status = 'interrupted', owner = NULL, owner_started = NULL
                                  WHERE id = ? AND status = 'running' AND attempt = ?""",
                               (row["id"], row["attempt"]))
                    if changed.rowcount:
                        db.execute("UPDATE attempts SET status = 'interrupted' WHERE job = ? AND number = ?",
                                   (row["id"], row["attempt"]))
            return [{**dict(row), "parameters": json.loads(row["parameters"])}
                    for row in db.execute("SELECT * FROM jobs ORDER BY rowid")]

    def start(self, identifier):
        import psutil

        with closing(self._connect()) as db, db:
            changed = db.execute("""UPDATE jobs SET status = 'running', attempt = attempt + 1,
                                    owner = ?, owner_started = ?, error = NULL
                                    WHERE id = ? AND status IN ('pending', 'cancelled', 'failed', 'interrupted', 'rejected')
                                    AND NOT EXISTS (SELECT 1 FROM jobs WHERE status = 'running')""",
                                 (os.getpid(), psutil.Process().create_time(), identifier))
            if changed.rowcount != 1:
                raise ValueError("El trabajo no está disponible o ya hay otro trabajo usando el motor.")
            number, parameters = db.execute("SELECT attempt, parameters FROM jobs WHERE id = ?",
                                            (identifier,)).fetchone()
            db.execute("INSERT INTO attempts (job, number, parameters, status) VALUES (?, ?, ?, 'running')",
                       (identifier, number, parameters))
            return number

    def transition(self, identifier, status, error=None, result=None, *, expected=None, attempt=None):
        previous = {
            "complete": ("running",), "failed": ("running",), "cancelled": ("pending", "running"),
            "accepted": ("complete",), "rejected": ("complete",),
        }[status]
        if expected is not None:
            if expected not in previous:
                raise ValueError("Transición de trabajo inválida.")
            previous = (expected,)
        placeholders = ", ".join("?" for _ in previous)
        with closing(self._connect()) as db, db:
            changed = db.execute(f"""UPDATE jobs SET status = ?, owner = NULL, owner_started = NULL,
                                     error = ?, result = COALESCE(?, result)
                                     WHERE id = ? AND status IN ({placeholders}) AND (? IS NULL OR attempt = ?)""",
                                 (status, error, result, identifier, *previous, attempt, attempt))
            if changed.rowcount != 1:
                raise ValueError("El estado del trabajo cambió. Actualiza la lista.")
            if previous != ("pending",):
                db.execute("""UPDATE attempts SET status = ?, error = COALESCE(?, error), result = COALESCE(?, result)
                              WHERE job = ? AND number = (SELECT attempt FROM jobs WHERE id = ?)
                              AND status != 'interrupted'""", (status, error, result, identifier, identifier))
