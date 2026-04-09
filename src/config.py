"""
src/config.py
-------------
Central configuration for Scout AI Interviewer.

Architecture:
  OpenAI GPT-4o-mini  → question embeddings + question selection + scoring + meta-evaluation

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

    # ── OpenAI — all roles ─────────────────────────────────────────
    # Single API key powers everything:
    #   1. Embeddings  (text-embedding-3-small) — ChromaDB indexing + query embedding
    #   2. Interviewer (OPENAI_INTERVIEWER_MODEL) — question selection, rephrasing, scoring
    #   3. Evaluator   (OPENAI_EVAL_MODEL) — meta-evaluation of interviewer quality
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Embedding model for ChromaDB indexing and query embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Parallel embedding workers
    EMBEDDING_WORKERS: int = int(os.getenv("EMBEDDING_WORKERS", "8"))

    # Interviewer model — question selection, rephrasing, scoring
    OPENAI_INTERVIEWER_MODEL: str = os.getenv("OPENAI_INTERVIEWER_MODEL", "gpt-4o-mini")
    OPENAI_INTERVIEWER_TEMPERATURE: float = float(os.getenv("OPENAI_INTERVIEWER_TEMPERATURE", "0.7"))

    # Evaluator model — meta-evaluation only
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