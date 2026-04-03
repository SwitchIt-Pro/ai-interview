"""
src/config.py
-------------
Configuration for Scout AI VectorDB Sync Service.

This service has ONE job: keep ChromaDB in sync with questions_store.xlsx
using Qwen (nomic-embed-text) via Ollama for embeddings.

Load order:  .env file → environment variables → defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """All runtime settings for the VectorDB sync service."""

    # ── Paths ──────────────────────────────────────────────────────
    PROJECT_ROOT: Path = PROJECT_ROOT

    EXCEL_FILE_PATH: Path = PROJECT_ROOT / os.getenv(
        "EXCEL_FILE_PATH", "data/questions_store.xlsx"
    )

    CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / os.getenv(
        "CHROMA_PERSIST_DIR", "chroma_store"
    )

    # ── ChromaDB ───────────────────────────────────────────────────
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "scout_questions"
    )

    # ── Ollama ─────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Embedding model via Ollama (nomic-embed-text is fast & purpose-built)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Parallel embedding workers — concurrent requests to Ollama
    EMBEDDING_WORKERS: int = int(os.getenv("EMBEDDING_WORKERS", "8"))

    # ── Sync settings ──────────────────────────────────────────────
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))

    # File watcher poll interval in seconds
    SYNC_POLL_INTERVAL: int = int(os.getenv("SYNC_POLL_INTERVAL", "10"))

    # ── Excel Sheet definition ─────────────────────────────────────
    SHEET_NAME: str = "Questions Store"

    # Column index (0-based) mapping
    COLUMNS = {
        "question_id":              0,
        "question_text":            1,
        "industry":                 2,
        "role":                     3,
        "evaluation_area":          4,
        "experience_level":         5,
        "priority_level":           6,
        "question_type":            7,
        "question_depth":           8,
        "skill_clusters_tested":    9,
        "what_ai_listens_for":     10,
        "strong_signal_example":   11,
        "weak_signal_example":     12,
        "follow_up_trigger":       13,
        "parent_question_id":      14,
        "status":                  15,
        "ctc_band":                16,
        "expected_duration_seconds": 17,
        "difficulty_score":        18,
        "usage_count":             19,
        "success_rate":            20,
        "variant_group_id":        21,
        "tags":                    22,
    }


# Singleton
config = Config()