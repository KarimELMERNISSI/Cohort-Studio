"""
LanceDB engine: embedded, serverless, columnar storage on disk (Lance format).

Vectors live in files under the user's workspace, so memory use stays low and nothing has to run
beside the application. Search is exact (brute force, cosine) up to :data:`ANN_THRESHOLD` chunks;
beyond that :meth:`LanceDBIndex.optimize` builds an IVF index for approximate search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.registry import Availability

from .base import BaseVectorIndex, ChunkRow, Hit, VectorEngine, VectorStoreError

logger = logging.getLogger(__name__)

TABLE = "chunks"
ANN_THRESHOLD = 50_000


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LanceDBIndex(BaseVectorIndex):
    engine = VectorEngine.LANCEDB
    name = "lancedb"
    display_name = "LanceDB"
    summary = "Embedded and serverless; vectors stored on disk in the columnar Lance format."

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.directory = directory
        self._db: Any = None

    @classmethod
    def availability(cls) -> Availability:
        return Availability.of_module("lancedb", extra="ai")

    @property
    def location(self) -> str:
        return str(self.directory)

    def _connect(self):
        if self._db is None:
            import lancedb

            self.directory.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.directory))
        return self._db

    def _tables(self) -> list[str]:
        listed = self._connect().list_tables()
        return list(getattr(listed, "tables", listed) or [])

    def exists(self) -> bool:
        return self.directory.exists() and TABLE in self._tables()

    def _table(self):
        if not self.exists():
            raise VectorStoreError("No LanceDB index yet: build it from your documents first.")
        return self._connect().open_table(TABLE)

    def reset(self, dim: int) -> None:
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("page", pa.int64()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), int(dim))),
            ]
        )
        db = self._connect()
        db.drop_table(TABLE, ignore_missing=True)
        db.create_table(TABLE, schema=schema)

    def add(self, rows: list[ChunkRow]) -> None:
        if not rows:
            return
        self._table().add(
            [
                {
                    "id": r.id,
                    "text": r.text,
                    "source": r.source,
                    "page": int(r.page),
                    "metadata": r.metadata_json(),
                    "vector": [float(v) for v in r.vector],
                }
                for r in rows
            ]
        )

    def query(self, vector: list[float], k: int, sources: list[str] | None = None) -> list[Hit]:
        import json

        builder = self._table().search([float(v) for v in vector]).distance_type("cosine")
        if sources:
            builder = builder.where(
                f"source IN ({', '.join(_sql_literal(s) for s in sources)})", prefilter=True
            )
        hits = []
        for row in builder.limit(int(k)).to_list():
            try:
                metadata = json.loads(row.get("metadata") or "{}")
            except ValueError:
                metadata = {}
            metadata.setdefault("source", row.get("source", ""))
            metadata.setdefault("page", row.get("page", -1))
            hits.append(
                Hit(
                    text=row.get("text", ""),
                    metadata=metadata,
                    score=1.0 - float(row.get("_distance", 1.0)),
                    vector=list(row["vector"]) if row.get("vector") is not None else None,
                )
            )
        return hits

    def count(self) -> int:
        return int(self._table().count_rows()) if self.exists() else 0

    def optimize(self) -> None:
        n = self.count()
        if n < ANN_THRESHOLD:
            return  # exact search is fast and exact at this size
        try:
            self._table().create_index(metric="cosine", vector_column_name="vector", replace=True)
        except Exception as exc:  # an approximate index is an optimisation, never a requirement
            logger.warning("LanceDB index creation skipped: %s", exc)

    def close(self) -> None:
        self._db = None


__all__ = ["ANN_THRESHOLD", "LanceDBIndex"]
