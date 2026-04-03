"""
src/sync_service.py
-------------------
VectorDB Sync Service — the ONLY service in scout_ai_interviewer.

Watches questions_store.xlsx for changes and automatically keeps
ChromaDB in sync:
  - New rows added    → embeds them via Ollama and inserts into ChromaDB
  - Rows removed      → deletes them from ChromaDB
  - No changes        → does nothing (idle)

Architecture:
  Excel file  →  file hash check (poll every N seconds)
                    ↓  (if changed)
              ExcelQuestionReader  →  diff IDs  →  Embedder (Ollama)  →  ChromaDB

Usage:
  python -m src.sync_service              # Run the watcher (loops forever)
  python -m src.sync_service --once       # Run a single sync pass and exit
  python -m src.sync_service --status     # Show current DB vs Excel status
"""

from __future__ import annotations

import hashlib
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Set

from .config import config
from .excel_reader import ExcelQuestionReader
from .embedder import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Fast MD5 hash of a file — used to detect Excel changes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_all_chroma_ids(store: VectorStore) -> Set[str]:
    """Fetch only the IDs from ChromaDB — no documents, no embeddings."""
    total = store.count
    if total == 0:
        return set()

    all_ids: Set[str] = set()
    offset = 0
    batch_size = 1000

    while offset < total:
        raw = store._collection.get(
            include=[],
            limit=batch_size,
            offset=offset,
        )
        batch_ids = raw.get("ids", [])
        if not batch_ids:
            break
        all_ids.update(batch_ids)
        offset += len(batch_ids)

    return all_ids


# ──────────────────────────────────────────────────────────────────
# Sync Logic
# ──────────────────────────────────────────────────────────────────

def run_sync(store: VectorStore, reader: ExcelQuestionReader) -> dict:
    """
    Single sync pass: compare Excel ↔ ChromaDB and reconcile.

    Returns a summary dict with counts of added, removed, unchanged, total.
    """
    # 1. Load all active questions from Excel
    questions = reader.load_all(active_only=True)
    excel_map = {q.question_id: q for q in questions}
    excel_ids = set(excel_map.keys())

    # 2. Get existing IDs from ChromaDB
    existing_ids = _get_all_chroma_ids(store)

    # 3. Compute diff
    new_ids     = excel_ids - existing_ids       # added to Excel → add to DB
    removed_ids = existing_ids - excel_ids       # removed from Excel → delete

    unchanged = len(existing_ids & excel_ids)

    summary = {
        "excel_rows": len(excel_ids),
        "chroma_docs": len(existing_ids),
        "new": len(new_ids),
        "removed": len(removed_ids),
        "unchanged": unchanged,
    }

    # 4. Nothing to do?
    if not new_ids and not removed_ids:
        return summary

    # 5. Delete removed rows
    if removed_ids:
        print(f"  🗑️  Deleting {len(removed_ids)} removed question(s)...")
        store._collection.delete(ids=list(removed_ids))
        print(f"     ✓ Deleted {len(removed_ids)} from ChromaDB")

    # 6. Embed + insert new rows
    if new_ids:
        new_questions = [excel_map[qid] for qid in sorted(new_ids)]
        print(f"  ➕ Embedding {len(new_questions)} new question(s) via Ollama...")
        total = store.ingest_questions(new_questions)
        print(f"     ✓ Added {total} to ChromaDB")

    summary["total_after"] = store.count
    return summary


def print_status(store: VectorStore, reader: ExcelQuestionReader) -> None:
    """Print current status: Excel rows vs ChromaDB documents."""
    questions = reader.load_all(active_only=True)
    excel_ids = {q.question_id for q in questions}
    existing_ids = _get_all_chroma_ids(store)

    new_ids     = excel_ids - existing_ids
    removed_ids = existing_ids - excel_ids
    unchanged   = existing_ids & excel_ids

    print(f"\n{'═' * 60}")
    print(f"  SCOUT AI — VECTORDB STATUS")
    print(f"{'═' * 60}")
    print(f"  Excel file    : {config.EXCEL_FILE_PATH}")
    print(f"  ChromaDB      : {config.CHROMA_PERSIST_DIR}")
    print(f"  Collection    : {config.CHROMA_COLLECTION_NAME}")
    print(f"  Embed model   : {config.EMBEDDING_MODEL} (via Ollama)")
    print(f"{'─' * 60}")
    print(f"  Excel rows (active)     : {len(excel_ids)}")
    print(f"  ChromaDB documents      : {len(existing_ids)}")
    print(f"{'─' * 60}")
    print(f"  ✅ In sync (unchanged)  : {len(unchanged)}")
    print(f"  ➕ Pending inserts      : {len(new_ids)}")
    print(f"  🗑️  Pending deletes     : {len(removed_ids)}")

    if not new_ids and not removed_ids:
        print(f"\n  ✅ ChromaDB is fully in sync with Excel.")
    else:
        print(f"\n  ⚠️  Out of sync! Run sync to reconcile.")
    print(f"{'═' * 60}\n")


# ──────────────────────────────────────────────────────────────────
# File Watcher Loop
# ──────────────────────────────────────────────────────────────────

def watch_and_sync(
    poll_interval: Optional[int] = None,
) -> None:
    """
    Watch the Excel file and auto-sync ChromaDB whenever it changes.

    Polls the file hash every `poll_interval` seconds.
    On change: runs a full diff sync (add new rows, delete removed rows).
    """
    interval = poll_interval or config.SYNC_POLL_INTERVAL

    print(f"\n{'═' * 60}")
    print(f"  SCOUT AI — VECTORDB SYNC SERVICE")
    print(f"  Watching: {config.EXCEL_FILE_PATH}")
    print(f"  Polling every {interval}s for changes")
    print(f"{'═' * 60}\n")

    # ── Verify prerequisites ────────────────────────────────────
    if not config.EXCEL_FILE_PATH.exists():
        print(f"  ❌ Excel file not found: {config.EXCEL_FILE_PATH}")
        sys.exit(1)

    embedder = Embedder()
    try:
        embedder.check_connection()
        print(f"  ✓ Ollama connected — model: {embedder.model_name}")
    except RuntimeError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    store = VectorStore(embedder=embedder)
    reader = ExcelQuestionReader()

    print(f"  ✓ ChromaDB ready — {store.count} existing documents")

    # ── Initial sync on startup ──────────────────────────────────
    print(f"\n  ── Initial sync ──")
    summary = run_sync(store, reader)
    _print_sync_summary(summary)

    # ── Watch loop ───────────────────────────────────────────────
    last_hash = _file_hash(config.EXCEL_FILE_PATH)
    running = True

    def _stop(sig, frame):
        nonlocal running
        print(f"\n\n  ⏹  Stopping sync service...")
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"\n  👁️  Watching for changes (Ctrl+C to stop)...\n")

    while running:
        time.sleep(interval)

        if not config.EXCEL_FILE_PATH.exists():
            continue

        current_hash = _file_hash(config.EXCEL_FILE_PATH)
        if current_hash != last_hash:
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] 📝 Excel file changed — syncing...")
            last_hash = current_hash

            # Small delay to let Excel finish writing
            time.sleep(1)

            try:
                reader = ExcelQuestionReader()
                summary = run_sync(store, reader)
                _print_sync_summary(summary)
            except Exception as e:
                print(f"  [{ts}] ❌ Sync error: {e}")
                logger.exception("Sync failed")

    print("  ✅ Sync service stopped.\n")


def run_once() -> None:
    """Run a single sync pass (no watching) and exit."""
    print(f"\n{'═' * 60}")
    print(f"  SCOUT AI — ONE-TIME SYNC")
    print(f"{'═' * 60}\n")

    if not config.EXCEL_FILE_PATH.exists():
        print(f"  ❌ Excel file not found: {config.EXCEL_FILE_PATH}")
        sys.exit(1)

    embedder = Embedder()
    try:
        embedder.check_connection()
        print(f"  ✓ Ollama connected — model: {embedder.model_name}")
    except RuntimeError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    store = VectorStore(embedder=embedder)
    reader = ExcelQuestionReader()

    print(f"  ✓ ChromaDB has {store.count} existing documents")
    print(f"  ── Syncing ──\n")

    summary = run_sync(store, reader)
    _print_sync_summary(summary)

    print(f"\n{'═' * 60}\n")


def _print_sync_summary(summary: dict) -> None:
    """Pretty-print the sync result."""
    print(f"\n  📊 Sync Result:")
    print(f"     Excel rows        : {summary['excel_rows']}")
    print(f"     ChromaDB docs     : {summary['chroma_docs']}")
    print(f"     Unchanged         : {summary['unchanged']}")
    print(f"     Added             : {summary['new']}")
    print(f"     Removed           : {summary['removed']}")

    if summary["new"] == 0 and summary["removed"] == 0:
        print(f"     ✅ Already in sync — no changes needed")
    else:
        total = summary.get("total_after", "?")
        print(f"     ✅ Synced! ChromaDB now has {total} documents")


# ──────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scout AI — VectorDB Sync Service (Excel ↔ ChromaDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single sync pass and exit (no watching).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current Excel vs ChromaDB status and exit.",
    )
    parser.add_argument(
        "--force-reload", action="store_true",
        help="Wipe ChromaDB and re-embed everything from scratch.",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help=f"Poll interval in seconds (default: {config.SYNC_POLL_INTERVAL}).",
    )
    args = parser.parse_args()

    if args.status:
        embedder = Embedder()
        store = VectorStore(embedder=embedder)
        reader = ExcelQuestionReader()
        print_status(store, reader)
        return

    if args.force_reload:
        _force_reload()
        return

    if args.once:
        run_once()
    else:
        watch_and_sync(poll_interval=args.interval)


def _force_reload():
    """Wipe ChromaDB and reload everything from Excel."""
    print(f"\n{'═' * 60}")
    print(f"  SCOUT AI — FORCE RELOAD")
    print(f"{'═' * 60}\n")

    embedder = Embedder()
    try:
        embedder.check_connection()
    except RuntimeError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    store = VectorStore(embedder=embedder)
    reader = ExcelQuestionReader()

    print(f"  ⚠️  Wiping {store.count} existing documents...")
    store.nuke_and_recreate()
    print(f"  ✓ ChromaDB cleared\n")

    questions = reader.load_all(active_only=True)
    print(f"  Embedding all {len(questions)} questions via Ollama...\n")
    total = store.ingest_questions(questions)

    print(f"\n  ✅ Done! {total} documents loaded into ChromaDB")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
