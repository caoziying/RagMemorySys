"""上传任务队列（SQLite 实现）。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


class UploadQueue:
    """基于 SQLite 的轻量上传队列，支持重试与 DLQ。"""

    def __init__(self, db_path: str | None = None, retry_limit: int = 5) -> None:
        base_dir = Path(settings.data_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(base_dir / "upload_queue.db")
        self.retry_limit = retry_limit
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS upload_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'queued',
                    next_attempt_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS upload_dlq (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    failed_at TEXT NOT NULL,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS processed_requests (
                    request_id TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    async def enqueue(self, request_id: str, payload: dict[str, Any]) -> bool:
        return await asyncio.to_thread(self._enqueue_sync, request_id, payload)

    def _enqueue_sync(self, request_id: str, payload: dict[str, Any]) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO upload_queue
                (request_id, payload, attempts, status, next_attempt_at, created_at, updated_at)
                VALUES (?, ?, 0, 'queued', ?, ?, ?)
                """,
                (request_id, json.dumps(payload, ensure_ascii=False), now, now, now),
            )
            return conn.total_changes > 0

    async def is_processed(self, request_id: str) -> bool:
        return await asyncio.to_thread(self._is_processed_sync, request_id)

    def _is_processed_sync(self, request_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            return row is not None

    async def mark_processed(self, request_id: str) -> None:
        await asyncio.to_thread(self._mark_processed_sync, request_id)

    def _mark_processed_sync(self, request_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_requests(request_id, processed_at) VALUES (?, ?)",
                (request_id, now),
            )

    async def get_next_message(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_next_message_sync)

    def _get_next_message_sync(self) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM upload_queue
                WHERE status='queued' AND next_attempt_at <= ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None

            conn.execute(
                "UPDATE upload_queue SET status='processing', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            conn.execute("COMMIT")
            return {
                "id": row["id"],
                "request_id": row["request_id"],
                "payload": json.loads(row["payload"]),
                "attempts": row["attempts"],
            }

    async def ack(self, msg_id: int) -> None:
        await asyncio.to_thread(self._ack_sync, msg_id)

    def _ack_sync(self, msg_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM upload_queue WHERE id=?", (msg_id,))

    async def fail(self, message: dict[str, Any], error: str) -> None:
        await asyncio.to_thread(self._fail_sync, message, error)

    def _fail_sync(self, message: dict[str, Any], error: str) -> None:
        msg_id = message["id"]
        attempts = int(message["attempts"]) + 1
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()

        with self._connect() as conn:
            if attempts >= self.retry_limit:
                row = conn.execute(
                    "SELECT request_id, payload FROM upload_queue WHERE id=?",
                    (msg_id,),
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        INSERT INTO upload_dlq (request_id, payload, attempts, failed_at, last_error)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (row["request_id"], row["payload"], attempts, now, error[:2000]),
                    )
                conn.execute("DELETE FROM upload_queue WHERE id=?", (msg_id,))
            else:
                backoff_seconds = min(2 ** attempts, 300)
                next_attempt = (now_dt + timedelta(seconds=backoff_seconds)).isoformat()
                conn.execute(
                    """
                    UPDATE upload_queue
                    SET status='queued', attempts=?, next_attempt_at=?, updated_at=?, last_error=?
                    WHERE id=?
                    """,
                    (attempts, next_attempt, now, error[:2000], msg_id),
                )
