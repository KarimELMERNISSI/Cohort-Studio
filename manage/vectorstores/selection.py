"""
Choosing and opening a vector database for the user's machine and needs.

- :class:`VectorStoreSettings` reads the server defaults (``CMP_VECTOR_STORE``, ``CMP_QDRANT_URL``,
  ``CMP_QDRANT_API_KEY``); the Settings page lets each user override the engine for the session.
- :func:`probe_machine` measures CPU cores, memory and free disk without extra dependencies.
- :func:`recommend` turns the machine, the size of the document library and the presence of a
  Qdrant server into a recommendation with its reasons and warnings (pure, unit-tested).
- :func:`select_engine` chooses the engine: the session's choice, the saved preference, the server's setting, then
  the recommendation (through the registry, :data:`manage.vectorstores.VECTOR_STORES`).
- :func:`open_index` opens the selected engine in the user's workspace.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from core.registry import AUTO, Selection
from manage.vectorstores import VECTOR_STORES
from utils.settings import Settings, SettingsError, get_settings

from .base import BaseVectorIndex, VectorEngine, VectorStoreError
from .lancedb_index import ANN_THRESHOLD

EMBEDDED_QDRANT_LIMIT = (
    20_000  # chunks: local mode keeps every vector in memory, brute-force search
)
DEFAULT_DIM = 768  # nomic-embed-text and most local embedding models
BYTES_PER_CHUNK_OF_PDF = (
    2_500  # rough: ~30% of a PDF's bytes is text, 1 000-character chunks with overlap
)


@dataclass(frozen=True)
class VectorStoreSettings:
    engine: str = AUTO  # auto | lancedb | qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)  # a key: never printed

    @classmethod
    def from_env(cls, settings: Settings | None = None) -> VectorStoreSettings:
        """The vector store's settings (utils.settings; the current ones by default)."""
        section = (settings or get_settings()).vector_store
        engine = section.engine or AUTO
        problem = VECTOR_STORES.setting_problem(engine)
        if problem:  # start-up stops on it first (manage.connectors); never the value: it can be a pasted key
            raise SettingsError(f"{problem}.")
        key = section.qdrant_api_key
        return cls(
            engine=engine,
            qdrant_url=section.qdrant_url,
            qdrant_api_key=key.get_secret_value() if key is not None else None,
        )


@dataclass(frozen=True)
class MachineProfile:
    cpu_count: int
    memory_gb: float | None  # total physical memory
    available_gb: float | None  # memory currently free
    disk_free_gb: float | None  # free space where the index is stored
    system: str = field(default_factory=platform.system)


def _memory_bytes() -> tuple[int | None, int | None]:
    """Total and available physical memory, without psutil."""
    try:
        if sys.platform == "win32":

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
            return None, None
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        try:
            available = page * os.sysconf("SC_AVPHYS_PAGES")
        except ValueError, OSError:
            available = None
        return int(total), (int(available) if available is not None else None)
    except AttributeError, ValueError, OSError:
        return None, None


def probe_machine(storage: Path | None = None) -> MachineProfile:
    total, available = _memory_bytes()
    disk = None
    if storage is not None:
        target = storage
        while not target.exists() and target.parent != target:
            target = target.parent
        try:
            disk = shutil.disk_usage(target).free / 1e9
        except OSError:
            disk = None
    gb = 1e9
    return MachineProfile(
        cpu_count=os.cpu_count() or 1,
        memory_gb=round(total / gb, 1) if total else None,
        available_gb=round(available / gb, 1) if available else None,
        disk_free_gb=round(disk, 1) if disk is not None else None,
    )


def estimate_chunks(pdf_sizes: list[int]) -> int:
    """Rough number of chunks a set of PDFs will produce (from their file sizes)."""
    return max(0, int(sum(pdf_sizes) / BYTES_PER_CHUNK_OF_PDF))


def estimate_index_mb(chunks: int, dim: int = DEFAULT_DIM) -> float:
    """Approximate size of an index: float32 vectors plus text and metadata."""
    return chunks * (dim * 4 + 1_500) / 1e6


@dataclass(frozen=True)
class Recommendation:
    engine: VectorEngine
    mode: str  # embedded | server
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def recommend(
    machine: MachineProfile,
    *,
    chunks: int,
    dim: int = DEFAULT_DIM,
    qdrant_server: bool = False,
) -> Recommendation:
    """The engine that suits this machine and library best, with the reasons."""
    size_mb = estimate_index_mb(chunks, dim)
    warnings: list[str] = []
    if machine.disk_free_gb is not None and machine.disk_free_gb * 1_000 < 2 * size_mb:
        warnings.append(
            f"Only {machine.disk_free_gb:.1f} GB free on disk for an index of about {size_mb:,.0f} MB."
        )
    if qdrant_server:
        return Recommendation(
            VectorEngine.QDRANT,
            "server",
            (
                "A Qdrant server is configured: one shared service with approximate (HNSW) search, payload "
                "indexes and concurrent sessions.",
                "Best for several users or very large libraries; this machine only sends queries.",
            ),
            tuple(warnings),
        )
    size = f" (index of about {size_mb:,.0f} MB for ~{chunks:,} chunks)" if chunks else ""
    reasons = [
        "Embedded: nothing to install or run beside the application.",
        f"Vectors stay on disk, so memory use stays low{size}.",
    ]
    if chunks >= ANN_THRESHOLD:
        reasons.append(
            "Exact search on small libraries; an approximate (IVF) index is built for large ones."
        )
    else:
        reasons.append(
            "Exact search at this size: every query compares against every chunk, with no approximation."
        )
    if machine.memory_gb is not None and machine.memory_gb < 8:
        reasons.append(f"Suits this machine's {machine.memory_gb:.0f} GB of memory.")
    return Recommendation(VectorEngine.LANCEDB, "embedded", tuple(reasons), tuple(warnings))


def qdrant_embedded_warnings(
    machine: MachineProfile, *, chunks: int, dim: int = DEFAULT_DIM
) -> list[str]:
    """Why embedded Qdrant may not suit this library or machine."""
    size_mb = estimate_index_mb(chunks, dim)
    notes = [
        "Embedded Qdrant keeps every vector in memory and opens a folder from one process at a time.",
    ]
    if chunks > EMBEDDED_QDRANT_LIMIT:
        notes.append(
            f"About {chunks:,} chunks is above the ~{EMBEDDED_QDRANT_LIMIT:,} chunks embedded mode suits: "
            "run a Qdrant server (docker run -p 6333:6333 qdrant/qdrant) and set CMP_QDRANT_URL, or use LanceDB."
        )
    available = machine.available_gb or machine.memory_gb
    if available is not None and size_mb > available * 1_000 * 0.25:
        notes.append(
            f"The index (~{size_mb:,.0f} MB) would take more than a quarter of the free memory "
            f"({available:.1f} GB)."
        )
    return notes


def select_engine(
    *,
    session: str | None = None,
    preference: str | None = None,
    settings: VectorStoreSettings,
    advice: Recommendation,
) -> Selection:
    """
    The engine to use: the session's choice, then the saved preference, then the server's setting, then the engine
    recommended for this machine (with its reasons), then the family's order. A choice that cannot run here falls
    through, and the selection says why.
    """
    return VECTOR_STORES.select(
        session=session,
        preference=preference,
        setting=settings.engine,
        auto=advice.engine.value,
        auto_reasons=advice.reasons,
    )


def open_index(
    engine: str,
    *,
    base_dir: Path,
    username: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> BaseVectorIndex:
    """Open the engine's index in ``base_dir/<engine>`` (on a server when the engine has one and a URL is given)."""
    cls = VECTOR_STORES.get(engine)
    found = VECTOR_STORES.availability(cls.name)
    if not found.ok:
        raise VectorStoreError(
            f"{cls.display_name} cannot be used: {found.reason}"
            + (f" ({found.install_hint})" if found.install_hint else "")
        )
    return cls.open(
        Path(base_dir) / cls.name, username=username, url=qdrant_url, api_key=qdrant_api_key
    )


__all__ = [
    "AUTO",
    "EMBEDDED_QDRANT_LIMIT",
    "MachineProfile",
    "Recommendation",
    "VectorStoreSettings",
    "estimate_chunks",
    "estimate_index_mb",
    "open_index",
    "probe_machine",
    "qdrant_embedded_warnings",
    "recommend",
    "select_engine",
]
