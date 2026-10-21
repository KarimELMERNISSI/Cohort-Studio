"""
Qdrant engine, in two modes.

- **Server** (``CMP_QDRANT_URL``, optional ``CMP_QDRANT_API_KEY``): a Qdrant service shared by every
  session, with HNSW approximate search, payload indexes and concurrent access. One collection
  per user (``cmp_<user>_chunks``). Best for several users or very large libraries.
- **Embedded** (no URL): Qdrant's local mode in the user's workspace. Every vector is held in
  memory and a folder can be opened by one client only, so the client is shared per folder in this
  process. Suited to small and medium libraries (tens of thousands of chunks).

Every rebuild writes a new collection (``<base>_<id>``) and then drops the previous one; the active
name is kept in ``collection.txt`` next to the manifest. Embedded Qdrant on Windows can reload a
deleted collection's files when the same name is recreated, so names are never reused.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from core.registry import Availability

from .base import BaseVectorIndex, ChunkRow, Hit, VectorEngine, VectorStoreError

LOCAL_COLLECTION = "chunks"
POINTER_NAME = "collection.txt"
_CLIENTS: dict[tuple[str, str, str], Any] = {}
_LOCK = threading.Lock()


def _client(*, path: Path | None = None, url: str | None = None, api_key: str | None = None):
    """One client per folder (local mode allows a single client) or per server and key."""
    from qdrant_client import QdrantClient

    key = ("path", str(path), "") if path is not None else ("url", str(url), api_key or "")
    with _LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(path))
            else:
                client = QdrantClient(url=url, api_key=api_key or None, timeout=10)
            _CLIENTS[key] = client
        return client


def release_clients() -> None:
    """Close every shared client (tests, shutdown)."""
    with _LOCK:
        for client in _CLIENTS.values():
            try:
                client.close()
            except Exception:
                pass  # best-effort clean-up: the client is dropped either way
        _CLIENTS.clear()


def user_collection(username: str | None) -> str:
    """Collection of a user on a shared Qdrant server."""
    safe = re.sub(r"[^a-z0-9_]+", "_", (username or "shared").lower()).strip("_") or "shared"
    return f"cmp_{safe}_chunks"


class QdrantIndex(BaseVectorIndex):
    engine = VectorEngine.QDRANT
    name = "qdrant"
    display_name = "Qdrant"
    summary = "Embedded in-memory for small libraries, or a Qdrant server for several users and large libraries."

    def __init__(
        self,
        manifest_dir: Path,
        *,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = LOCAL_COLLECTION,
    ) -> None:
        super().__init__(manifest_dir)
        if path is None and not url:
            raise VectorStoreError("Qdrant needs a local folder (embedded mode) or a server URL.")
        self.path = path
        self.url = url
        self.api_key = api_key
        self.collection = collection

    @classmethod
    def availability(cls) -> Availability:
        return Availability.of_module("qdrant_client", extra="ai")

    @property
    def mode(self) -> str:
        return "server" if self.url else "embedded"

    @property
    def location(self) -> str:
        return f"{self.url} ({self.active_collection})" if self.url else str(self.path)

    def _client(self):
        try:
            if self.url:
                return _client(url=self.url, api_key=self.api_key)
            return _client(path=self.path)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant could not be opened at {self.location}: {exc}") from exc

    def ping(self) -> tuple[bool, str]:
        """Reachability of the server (or the local folder)."""
        try:
            collections = self._client().get_collections().collections
            return True, f"Qdrant reachable ({len(collections)} collection(s))."
        except Exception as exc:
            return False, f"Qdrant is not reachable at {self.location}: {exc}"

    @property
    def active_collection(self) -> str:
        """Collection holding the current index (the base name for indexes built before generations)."""
        pointer = self.manifest_dir / POINTER_NAME
        try:
            name = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
        except OSError:
            name = ""
        return name or self.collection

    def exists(self) -> bool:
        try:
            return bool(self._client().collection_exists(self.active_collection))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant error at {self.location}: {exc}") from exc

    def reset(self, dim: int) -> None:
        from qdrant_client import models

        client = self._client()
        previous = self.active_collection
        name = f"{self.collection}_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=int(dim), distance=models.Distance.COSINE),
        )
        if self.mode == "server":  # payload indexes have no effect in local mode
            client.create_payload_index(
                name, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
            )
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        (self.manifest_dir / POINTER_NAME).write_text(name, encoding="utf-8")
        if previous != name and client.collection_exists(previous):
            client.delete_collection(previous)

    def add(self, rows: list[ChunkRow]) -> None:
        if not rows:
            return
        from qdrant_client import models

        self._client().upsert(
            self.active_collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, r.id)),
                    vector=[float(v) for v in r.vector],
                    payload={
                        "text": r.text,
                        "source": r.source,
                        "page": int(r.page),
                        "metadata": r.metadata_json(),
                    },
                )
                for r in rows
            ],
            wait=True,
        )

    def query(self, vector: list[float], k: int, sources: list[str] | None = None) -> list[Hit]:
        from qdrant_client import models

        query_filter = None
        if sources:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchAny(any=list(sources)))]
            )
        points = (
            self._client()
            .query_points(
                self.active_collection,
                query=[float(v) for v in vector],
                limit=int(k),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=True,
            )
            .points
        )
        hits = []
        for point in points:
            payload = point.payload or {}
            try:
                metadata = json.loads(payload.get("metadata") or "{}")
            except ValueError:
                metadata = {}
            metadata.setdefault("source", payload.get("source", ""))
            metadata.setdefault("page", payload.get("page", -1))
            vec = point.vector if isinstance(point.vector, list) else None
            hits.append(
                Hit(
                    text=payload.get("text", ""),
                    metadata=metadata,
                    score=float(point.score),
                    vector=vec,
                )
            )
        return hits

    def count(self) -> int:
        if not self.exists():
            return 0
        return int(self._client().count(self.active_collection, exact=True).count)


__all__ = ["LOCAL_COLLECTION", "QdrantIndex", "release_clients", "user_collection"]
