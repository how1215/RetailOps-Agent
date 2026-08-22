import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceStore:
    """Small JSONL trace sink that keeps portfolio runs reproducible and inspectable."""

    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def record(self, trace_id: str, event: str, **attributes: Any) -> None:
        item = {
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": trace_id,
            "event": event,
            **attributes,
        }
        with self._lock:
            self._events.setdefault(trace_id, []).append(item)
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with (self.trace_dir / f"{trace_id}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def get(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(trace_id, []))
