"""
Centralized Data Path Configuration

All data paths are configured relative to a single DATA_ROOT environment variable.
This allows Docker deployments to use mounted volumes while keeping local development simple.

Usage:
    from utils.data_paths import get_datasets_dir, get_chroma_dir

Environment Variables:
    DATA_ROOT: Base directory for all data storage (default: "./data")
"""

import os

# Base data directory - configurable via environment variable
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")


def get_data_root() -> str:
    """Get the base data directory."""
    return DATA_ROOT


def get_datasets_dir() -> str:
    """Get the datasets directory path."""
    path = os.path.join(DATA_ROOT, "datasets")
    os.makedirs(path, exist_ok=True)
    return path


def get_stats_dir() -> str:
    """Get the statistics cache directory path."""
    path = os.path.join(DATA_ROOT, "stats")
    os.makedirs(path, exist_ok=True)
    return path


def get_chroma_dir() -> str:
    """Get the ChromaDB vector database directory path."""
    path = os.path.join(DATA_ROOT, "chroma_db")
    os.makedirs(path, exist_ok=True)
    return path


def get_traces_dir() -> str:
    """Get the analysis traces directory path."""
    path = os.path.join(DATA_ROOT, "traces")
    os.makedirs(path, exist_ok=True)
    return path


def get_uploads_dir() -> str:
    """Get the uploads directory path."""
    path = os.path.join(DATA_ROOT, "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def get_taxonomy_dir() -> str:
    """Get the taxonomy directory path."""
    path = os.path.join(DATA_ROOT, "taxonomy")
    os.makedirs(path, exist_ok=True)
    return path


def get_enrichment_dir() -> str:
    """Get the enrichment input directory path."""
    path = os.path.join(DATA_ROOT, "enrichment")
    os.makedirs(path, exist_ok=True)
    return path


def get_enriched_dir() -> str:
    """Get the enriched output directory path."""
    path = os.path.join(DATA_ROOT, "enriched")
    os.makedirs(path, exist_ok=True)
    return path


def get_comparisons_dir() -> str:
    """Get the comparisons directory path."""
    path = os.path.join(DATA_ROOT, "comparisons")
    os.makedirs(path, exist_ok=True)
    return path


def get_knowledge_graphs_dir() -> str:
    """Get the knowledge graphs directory path."""
    path = os.path.join(DATA_ROOT, "knowledge_graphs")
    os.makedirs(path, exist_ok=True)
    return path


def get_rag_eval_logs_dir() -> str:
    """Get the RAG evaluation logs directory path."""
    path = os.path.join(DATA_ROOT, "rag_eval_logs")
    os.makedirs(path, exist_ok=True)
    return path
