"""
scripts/load_data.py
--------------------
Load questions from Excel into ChromaDB using Qwen-7B embeddings.

Modes:
  python scripts/load_data.py            → Full load (first time)
  python scripts/load_data.py --sync     → Incremental: only add NEW rows,
                                           delete REMOVED rows, skip unchanged
  python scripts/load_data.py --force-reload  → Wipe ChromaDB and reload all
  python scripts/load_data.py --summary  → Show Excel stats, no loading

--sync is what you use after adding/removing rows in the Excel file.
It only re-embeds the rows that changed — skips all 3,691 unchanged rows.
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.excel_reader import ExcelQuestionReader
from src.embedder import Embedder
from src.vector_store import VectorStore


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scout AI — Excel → ChromaDB loader with incremental sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Incremental sync: only embed NEW questions added to Excel, "
            "delete questions removed from Excel. Skips unchanged rows entirely."
        ),
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Wipe ChromaDB collection then reload all questions fresh.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete (wipe) the entire ChromaDB collection and exit. Does NOT reload.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show Excel statistics only — do not load into ChromaDB.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel Qwen embedding threads (default: EMBEDDING_WORKERS in .env). "
            "Try 4 for CPU, 8 for GPU."
        ),
    )
    args = parser.parse_args()

    # ── Delete-only mode (wipe and exit) ──────────────────────────
    if args.delete:
        _delete_collection()
        return

    print("\n" + "═" * 65)
    print("  SCOUT AI — DATA LOADER")
    print("  Excel → OpenAI Embeddings → ChromaDB")
    print("═" * 65 + "\n")

    # ── 1. Read Excel ─────────────────────────────────────────────
    print("  [1/4] Reading Excel question store...")
    try:
        reader = ExcelQuestionReader()
    except FileNotFoundError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    if args.summary:
        _print_summary(reader)
        return

    questions = reader.load_all(active_only=True)
    print(f"       ✓ {len(questions)} active questions loaded from Excel")

    # ── 2. Verify OpenAI Embedder ─────────────────────────────────
    print("\n  [2/4] Verifying OpenAI Embedder connection...")
    embedder = Embedder(workers=args.workers)  # None → uses .env EMBEDDING_WORKERS
    try:
        embedder.check_connection()
        print(f"       ✓ OpenAI Embedder ready  |  model: {embedder.model_name}  |  workers: {embedder._workers}")
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    # ── 3. Open ChromaDB ──────────────────────────────────────────
    print("\n  [3/4] Opening ChromaDB...")
    store = VectorStore(embedder=embedder)

    if args.force_reload:
        print(f"       ⚠️  Force reload — wiping {store.count} existing documents...")
        store.nuke_and_recreate()
        print("       ✓ Collection cleared and ready")
    else:
        print(f"       ✓ Collection has {store.count} existing documents")

    # ── 4. Sync or Full Load ──────────────────────────────────────
    print("\n  [4/4] Loading data into ChromaDB...\n")

    if args.sync and store.is_populated:
        _run_incremental_sync(store, questions)
    else:
        # Full load (first time, or after --force-reload)
        print(f"  Embedding all {len(questions)} questions with OpenAI text-embedding-3-small...")
        print(f"  (This will call the OpenAI Embeddings API in parallel)\n")
        total = store.ingest_questions(questions)
        print(f"\n  ✅ Done! {total} documents loaded into ChromaDB")
        print(f"     Total in collection: {store.count}")

    print("\n" + "═" * 65)
    print("  Next step: python scripts/run_interview.py")
    print("═" * 65 + "\n")


# ──────────────────────────────────────────────────────────────────
# Incremental Sync
# ──────────────────────────────────────────────────────────────────

def _run_incremental_sync(store: VectorStore, excel_questions: list) -> None:
    """
    Sync ChromaDB with the current Excel file.

    Algorithm:
      1. Get all IDs currently in ChromaDB  (fast — no embeddings fetched)
      2. Get all IDs from Excel
      3. NEW      = Excel IDs not in ChromaDB  → embed + upsert only these
      4. REMOVED  = ChromaDB IDs not in Excel  → delete from ChromaDB
      5. UNCHANGED = in both                   → skip entirely

    Result: ChromaDB mirrors the Excel file exactly, with minimal work.
    """
    # ── Get existing IDs from ChromaDB (IDs only, no data) ────────
    print("  Fetching existing IDs from ChromaDB...", end=" ", flush=True)
    existing_ids = _get_all_chroma_ids(store)
    print(f"✓  ({len(existing_ids)} documents found)")

    # ── Build lookup from Excel ────────────────────────────────────
    excel_map = {q.question_id: q for q in excel_questions}
    excel_ids = set(excel_map.keys())

    # ── Compute diff ───────────────────────────────────────────────
    new_ids     = excel_ids - existing_ids      # added to Excel → add to DB
    removed_ids = existing_ids - excel_ids      # removed from Excel → delete from DB
    unchanged   = existing_ids & excel_ids      # in both → skip

    # ── Print diff summary ─────────────────────────────────────────
    print(f"\n  📊 Sync Diff:")
    print(f"     Excel rows         : {len(excel_ids)}")
    print(f"     ChromaDB documents : {len(existing_ids)}")
    print(f"     ────────────────────────────────────")
    print(f"     ✅ Unchanged        : {len(unchanged):>5}  (skipped)")
    print(f"     ➕ New rows         : {len(new_ids):>5}  (will embed + add)")
    print(f"     ❌ Removed rows     : {len(removed_ids):>5}  (will delete)")

    # ── Nothing to do ──────────────────────────────────────────────
    if not new_ids and not removed_ids:
        print(f"\n  ✅ ChromaDB is already in sync with Excel. No changes needed.")
        return

    # ── Delete removed questions ───────────────────────────────────
    if removed_ids:
        print(f"\n  Deleting {len(removed_ids)} removed question(s)...")
        _delete_from_store(store, list(removed_ids))
        print(f"  ✓ Deleted {len(removed_ids)} question(s) from ChromaDB")

    # ── Embed and add new questions ────────────────────────────────
    if new_ids:
        new_questions = [excel_map[qid] for qid in sorted(new_ids)]
        print(f"\n  Embedding {len(new_questions)} new question(s) with OpenAI Embedder...")
        print(f"  (Skipping {len(unchanged)} unchanged rows — no re-embedding needed)\n")
        total = store.ingest_questions(new_questions)
        print(f"\n  ✓ Added {total} new question(s) to ChromaDB")

    print(f"\n  ✅ Sync complete! ChromaDB now has {store.count} questions")


def _get_all_chroma_ids(store: VectorStore) -> set:
    """
    Fetch only the IDs from ChromaDB — no documents, no embeddings.
    This is very fast regardless of collection size.
    """
    total = store.count
    if total == 0:
        return set()

    # Fetch in batches to handle very large collections safely
    all_ids = set()
    offset = 0
    batch_size = 1000

    while offset < total:
        raw = store._collection.get(
            include=[],              # IDs only — no embeddings, no docs
            limit=batch_size,
            offset=offset,
        )
        batch_ids = raw.get("ids", [])
        if not batch_ids:
            break
        all_ids.update(batch_ids)
        offset += len(batch_ids)

    return all_ids


def _delete_from_store(store: VectorStore, ids: list) -> None:
    """Delete questions from ChromaDB by ID."""
    store._collection.delete(ids=ids)


# ──────────────────────────────────────────────────────────────────
# Summary helper
# ──────────────────────────────────────────────────────────────────

def _print_summary(reader: ExcelQuestionReader) -> None:
    """Print Excel statistics to terminal."""
    summary = reader.get_summary()
    print(f"\n  Excel Summary:")
    print(f"    Total rows  : {summary['total']}")
    print(f"    Active rows : {summary['active']}")
    print(f"\n    By Role (top 10):")
    for role, count in sorted(summary["by_role"].items(), key=lambda x: -x[1])[:10]:
        print(f"      {role:<35} : {count}")
    print(f"\n    By Question Type:")
    for qtype, count in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
        print(f"      {qtype:<35} : {count}")
    print(f"\n    By Experience Level:")
    for level, count in sorted(summary["by_level"].items(), key=lambda x: -x[1]):
        print(f"      {level:<35} : {count}")


# ──────────────────────────────────────────────────────────────────
# Delete Collection (--delete flag)
# ──────────────────────────────────────────────────────────────────

def _delete_collection() -> None:
    """
    Wipe the entire ChromaDB collection and exit.
    Does NOT reload any data — leaves ChromaDB empty.

    Usage:
      python scripts/load_data.py --delete

    After deleting, re-run without flags to reload:
      python scripts/load_data.py
    """
    from src.config import config

    print("\n" + "═" * 65)
    print("  SCOUT AI — DELETE CHROMADB COLLECTION")
    print("═" * 65)
    print(f"\n  Collection : {config.CHROMA_COLLECTION_NAME}")
    print(f"  Location   : {config.CHROMA_PERSIST_DIR}\n")

    # Open store (no embedder needed for delete)
    from src.embedder import Embedder
    from src.vector_store import VectorStore

    embedder = Embedder()
    store = VectorStore(embedder=embedder)

    count = store.count
    if count == 0:
        print("  ℹ️  Collection is already empty. Nothing to delete.\n")
        return

    confirm = input(f"  ⚠️  This will permanently delete {count} documents. Confirm? [yes/no]: ").strip().lower()
    if confirm not in ("yes", "y"):
        print("\n  Cancelled — nothing deleted.\n")
        return

    store.nuke_and_recreate()
    print(f"\n  ✅ Collection deleted. ChromaDB is now empty.")
    print(f"\n  To reload: python scripts/load_data.py")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
