from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SignalsStore:
    db_path: str

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    score REAL NOT NULL,
                    factors_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    async def save_signal(self, *, query: str, signal_payload: dict[str, Any]) -> None:
        await asyncio.to_thread(self._save_signal_sync, query, signal_payload)

    def _save_signal_sync(self, query: str, signal_payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO signals (timestamp, query, signal, score, factors_json, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    str(signal_payload.get("signal", "HOLD")),
                    float(signal_payload.get("score", 0.0)),
                    json.dumps(signal_payload.get("factors", {}), ensure_ascii=False),
                    json.dumps(signal_payload, ensure_ascii=False),
                ),
            )
            conn.commit()
