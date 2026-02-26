"""
src/config.py
-------------
Central configuration for Scout AI Interviewer.

Architecture:
  Qwen-7B via Ollama  → question embeddings + question generation (interviewer)
  OpenAI GPT-4o-mini  → meta-evaluator only (evaluates Qwen's questions/scoring)

Load order:  .env file → environment variables → defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """All runtime settings in one place."""

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

    # ── Ollama / Qwen-7B ───────────────────────────────────────────
    # Qwen is used strictly as the INTERVIEWER (question selection + scoring).
    # nomic-embed-text is used for EMBEDDINGS (fast, dedicated embedding model).
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    QWEN_MODEL: str      = os.getenv("QWEN_MODEL", "qwen2.5:7b")
    QWEN_TEMPERATURE: float = float(os.getenv("QWEN_TEMPERATURE", "0.7"))

    # Embedding model — separate from Qwen (nomic-embed-text is ~10-20x faster).
    # Used ONLY for ChromaDB indexing and query embedding.
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Parallel embedding workers — how many concurrent requests to Ollama.
    # 8 = fast for nomic-embed-text (lightweight model handles concurrency well).
    EMBEDDING_WORKERS: int = int(os.getenv("EMBEDDING_WORKERS", "8"))

    # ── OpenAI — meta-evaluator ONLY ──────────────────────────────
    # OpenAI evaluates:
    #   1. Whether Qwen's selected questions were appropriate
    #   2. Whether Qwen's scoring of responses was accurate/fair
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EVAL_MODEL: str = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini")
    OPENAI_EVAL_TEMPERATURE: float = float(os.getenv("OPENAI_EVAL_TEMPERATURE", "0.2"))

    # ── Interview / RAG settings ───────────────────────────────────
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "5"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
    TOP_K: int = int(os.getenv("TOP_K", "8"))

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
