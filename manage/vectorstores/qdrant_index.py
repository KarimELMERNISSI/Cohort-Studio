"""
Qdrant engine, in two modes.

- **Server** (``CMP_QDRANT_URL``, optional ``CMP_QDRANT_API_KEY``): a Qdrant service shared by every
  session, with HNSW approximate search, payload indexes and concurrent access. One collection
  per user, named at random (``cmp_<32 hexadecimal digits>``), never after the account: the server
  may belong to someone else (Qdrant Cloud). Best for several users or very large libraries.
- **Embedded** (no URL): Qdrant's local mode in the user's workspace. Every vector is held in
  memory and a folder can be opened by one client only, so the client is shared per folder in this
  process. Suited to small and medium libraries (tens of thousands of chunks).

Every rebuild writes a new collection and then drops the previous one; the active name is kept in
the user's workspace, in ``collection.txt`` next to the index's manifest. Each place has its own
folder for both: the embedded index the engine's folder, each server ``servers/<key>`` (the key is a
hash of its address as the client reads it), so that building in one place never loses, nor is
taken for, the index of another. Embedded Qdrant on Windows can reload a deleted collection's files
when the same name is recreated, so names are never reused.

Earlier versions kept one manifest and one pointer for every place, and named a server collection
after the account (``cmp_<user>_chunks``, then ``cmp_<user>_chunks_<id>``). Opening the engine moves
such an index's manifest and pointer to its server's folder (the manifest names the server); it is
never reused there, and the next rebuild on its server deletes it.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from core.registry import Availability

from .base import (
    MANIFEST_NAME,
    BaseVectorIndex,
    ChunkRow,
    Hit,
    IndexManifest,
    VectorEngine,
    VectorStoreError,
)

LOCAL_COLLECTION = "chunks"
POINTER_NAME = "collection.txt"  # in each place's folder: the collection in use
SERVERS = "servers"  # the folder of the servers' folders, in the engine's
SERVER_PREFIX = "cmp_"  # every collection on a server, of every version
DEFAULT_PORT = 6333  # the client's, for an address that gives none
# a name no collection has (a new one's is a random UUID's): this workspace has built nothing on the server
NO_INDEX = SERVER_PREFIX + "0" * 32
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


def _read(pointer: Path) -> str:
    """The collection a pointer file names ("" when there is none)."""
    try:
        return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
    except OSError:
        return ""


def explicit_address(url: str | None, api_key: str | None = None) -> str | None:
    """
    An address with the scheme the client gives one written without (https with an API key, http without), so that
    the address badge, the server's folder and the recipient read the server the client connects to.
    """
    text = (url or "").strip()
    if not text or "//" in text:
        return text or url
    return f"{'https' if api_key else 'http'}://{text}"


def _server_key(url: str) -> str:
    """An address as the client reads it (http when no scheme is given, the host in lower case, port 6333 when none
    is given, the path without a final slash): two spellings of one server share one folder."""
    text = url.strip().replace("\\", "/")
    parts = urlparse(text if "//" in text else f"http://{text}")
    host, port, path = (
        (parts.hostname or "").lower(),
        parts.port or DEFAULT_PORT,
        parts.path.rstrip("/"),
    )
    return f"{(parts.scheme or 'http').lower()}://{host}:{port}{path}"


def server_folder(directory: Path, url: str) -> Path:
    """The folder that keeps the manifest and the collection's name of one server's index, in the engine's folder."""
    return directory / SERVERS / hashlib.sha256(_server_key(url).encode("utf-8")).hexdigest()[:16]


def _built_on_server(manifest: IndexManifest | None) -> tuple[str, str] | None:
    """The server and the collection a manifest's index was built on, read from its location ("<url> (<name>)"); None
    for an embedded index."""
    location = manifest.location if manifest is not None else ""
    url, separator, rest = location.rpartition(" (")
    collection = rest[:-1] if rest.endswith(")") else ""
    if not separator or not url or not collection.startswith(SERVER_PREFIX):
        return None
    return url, collection


def adopt_earlier_index(directory: Path) -> None:
    """
    An earlier version kept one manifest and one pointer for every place, in ``directory``, both written by the last
    build. One built on a server moves to that server's folder (the manifest names the server), so that a build
    elsewhere neither loses it nor is taken for it. Reads and writes the workspace only.
    """
    manifest = IndexManifest.load(directory)
    built = _built_on_server(manifest)
    if manifest is None or built is None:
        return
    url, collection = built
    target = server_folder(directory, url)
    if not (target / MANIFEST_NAME).exists():
        manifest.save(target)
        (target / POINTER_NAME).write_text(collection, encoding="utf-8")
    (directory / MANIFEST_NAME).unlink(missing_ok=True)
    if _read(directory / POINTER_NAME).startswith(SERVER_PREFIX):
        (directory / POINTER_NAME).unlink(missing_ok=True)


def unrecorded_collection(directory: Path) -> str | None:
    """
    A server collection an earlier version's pointer names without the manifest that names its server (a build that
    stopped half-way): which server holds it cannot be known, so it can only be reported.
    """
    name = _read(directory / POINTER_NAME)
    return (
        name if name.startswith(SERVER_PREFIX) and IndexManifest.load(directory) is None else None
    )


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
    """The name earlier versions gave a user's collection on a server, after the account: kept to find an index built
    then (:attr:`QdrantIndex.named_after_account`), which the next rebuild replaces."""
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

    @classmethod
    def open(
        cls,
        directory: Path,
        *,
        username: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> QdrantIndex:
        adopt_earlier_index(directory)  # the workspace only: no request is made
        if url:  # a server shared by every session: one collection per user, its manifest in the server's folder
            return cls(
                server_folder(directory, url),
                url=url,
                api_key=api_key,
                collection=user_collection(username),
            )
        return cls(directory, path=directory / "storage")

    @property
    def mode(self) -> str:
        return "server" if self.url else "embedded"

    def _generation_name(self) -> str:
        """The name of a new collection: at random on a server (never the account), ``chunks_<id>`` embedded."""
        if self.url:
            return f"cmp_{uuid.uuid4().hex}"
        return f"{self.collection}_{uuid.uuid4().hex[:8]}"

    @property
    def remote(self) -> bool:
        return self.mode == "server"

    @property
    def _pointer(self) -> Path:
        """The file naming the collection in use, in this place's folder."""
        return self.manifest_dir / POINTER_NAME

    def _pointed(self) -> str:
        """The collection the pointer names ("" when none; embedded, never a server's, as an earlier version wrote)."""
        name = _read(self._pointer)
        return "" if not self.url and name.startswith(SERVER_PREFIX) else name

    @property
    def named_after_account(self) -> bool:
        """Whether the index in use on a server was built by an earlier version, under the account's name (read from
        the workspace: no request is made). It is never reused: its rebuild deletes it."""
        if not self.url or not (self._pointed() or self.manifest() is not None):
            return False
        active = self.active_collection
        return active == self.collection or active.startswith(f"{self.collection}_")

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
        """
        Collection holding the current index: the one the pointer names; else the base name of the earliest indexes
        (before generations), on a server only when this workspace built one there (two installations can share a
        server and a user name); else ``NO_INDEX``.
        """
        pointed = self._pointed()
        if pointed:
            return pointed
        if self.url and self.manifest() is None:
            return NO_INDEX
        return self.collection

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
        # the collection in use, one named after the account included: deleted once the new one exists
        previous = self.active_collection
        name = self._generation_name()
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=int(dim), distance=models.Distance.COSINE),
        )
        if self.mode == "server":  # payload indexes have no effect in local mode
            client.create_payload_index(
                name, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
            )
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._pointer.write_text(name, encoding="utf-8")
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


__all__ = [
    "LOCAL_COLLECTION",
    "NO_INDEX",
    "QdrantIndex",
    "adopt_earlier_index",
    "explicit_address",
    "release_clients",
    "server_folder",
    "unrecorded_collection",
    "user_collection",
]
