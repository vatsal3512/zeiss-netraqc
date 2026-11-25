"""Inspection log in SQLite."""
import sqlite3
from datetime import datetime

COLS = ["time", "type", "file", "score", "threshold", "verdict", "ms", "label"]


def init(path):
    with sqlite3.connect(path) as c:
        c.execute("create table if not exists log (id integer primary key, time text, type text, file text, "
                  "score real, threshold real, verdict text, ms real, label text)")


def add(path, row):
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
    with sqlite3.connect(path) as c:
        c.execute(f"insert into log ({','.join(COLS)}) values ({','.join('?' * len(COLS))})", [row.get(k) for k in COLS])


def rows(path, limit=200):
    with sqlite3.connect(path) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("select * from log order by id desc limit ?", (limit,))]


def counts(path):
    with sqlite3.connect(path) as c:
        total, fails = c.execute("select count(*), coalesce(sum(verdict = 'FAIL'), 0) from log").fetchone()
    return {"total": total, "fail": fails, "pass": total - fails}
