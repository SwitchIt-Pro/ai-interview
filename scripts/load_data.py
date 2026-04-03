"""
scripts/load_data.py
--------------------
One-time data loader: Excel → ChromaDB via Ollama embeddings.

For ongoing sync, use the sync service instead:
  python -m src.sync_service

Modes:
  python scripts/load_data.py                → Full load (first time)
  python scripts/load_data.py --sync         → Incremental sync (add new, delete removed)
  python scripts/load_data.py --force-reload → Wipe ChromaDB and reload all
  python scripts/load_data.py --delete       → Wipe ChromaDB and exit
  python scripts/load_data.py --summary      → Show Excel stats only
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


def main():
    parser = argparse.ArgumentParser(
        description="Scout AI — Excel → ChromaDB loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sync", action="store_true", help="Incremental sync only.")
    parser.add_argument("--force-reload", action="store_true", help="Wipe and reload all.")
    parser.add_argument("--delete", action="store_true", help="Wipe collection and exit.")
    parser.add_argument("--summary", action="store_true", help="Show Excel stats only.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel embedding threads.")
    args = parser.parse_args()

    if args.delete:
        _delete_collection()
        return

    print("\n" + "═" * 60)
    print("  SCOUT AI — DATA LOADER")
    print("  Excel → Ollama Embeddings → ChromaDB")
    print("═" * 60 + "\n")

    # 1. Read Excel
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

    # 2. Verify Ollama
    print("\n  [2/4] Verifying Ollama embedding model...")
    embedder = Embedder(workers=args.workers)
    try:
        embedder.check_connection()
        print(f"       ✓ {embedder.model_name} ready  |  workers: {embedder._workers}")
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    # 3. Open ChromaDB
    print("\n  [3/4] Opening ChromaDB...")
    store = VectorStore(embedder=embedder)

    if args.force_reload:
        print(f"       ⚠️  Force reload — wiping {store.count} existing documents...")
        store.nuke_and_recreate()
        print("       ✓ Collection cleared")
    else:
        print(f"       ✓ Collection has {store.count} existing documents")

    # 4. Sync or Full Load
    print("\n  [4/4] Loading data into ChromaDB...\n")

    if args.sync and store.is_populated:
        _run_incremental_sync(store, questions)
    else:
        print(f"  Embedding all {len(questions)} questions via Ollama...\n")
        total = store.ingest_questions(questions)
        print(f"\n  ✅ Done! {total} documents loaded into ChromaDB")
        print(f"     Total in collection: {store.count}")

    print("\n" + "═" * 60 + "\n")


def _run_incremental_sync(store, excel_questions):
    """Sync ChromaDB with Excel — add new, delete removed, skip unchanged."""
    print("  Fetching existing IDs from ChromaDB...", end=" ", flush=True)
    existing_ids = _get_all_chroma_ids(store)
    print(f"✓  ({len(existing_ids)} found)")

    excel_map = {q.question_id: q for q in excel_questions}
    excel_ids = set(excel_map.keys())

    new_ids     = excel_ids - existing_ids
    removed_ids = existing_ids - excel_ids
    unchanged   = existing_ids & excel_ids

    print(f"\n  📊 Sync Diff:")
    print(f"     Excel rows         : {len(excel_ids)}")
    print(f"     ChromaDB documents : {len(existing_ids)}")
    print(f"     ✅ Unchanged        : {len(unchanged):>5}  (skipped)")
    print(f"     ➕ New rows         : {len(new_ids):>5}  (will embed + add)")
    print(f"     ❌ Removed rows     : {len(removed_ids):>5}  (will delete)")

    if not new_ids and not removed_ids:
        print(f"\n  ✅ ChromaDB is already in sync. No changes needed.")
        return

    if removed_ids:
        print(f"\n  Deleting {len(removed_ids)} removed question(s)...")
        store._collection.delete(ids=list(removed_ids))
        print(f"  ✓ Deleted {len(removed_ids)} from ChromaDB")

    if new_ids:
        new_questions = [excel_map[qid] for qid in sorted(new_ids)]
        print(f"\n  Embedding {len(new_questions)} new question(s) via Ollama...\n")
        total = store.ingest_questions(new_questions)
        print(f"\n  ✓ Added {total} new question(s)")

    print(f"\n  ✅ Sync complete! ChromaDB now has {store.count} questions")


def _get_all_chroma_ids(store):
    total = store.count
    if total == 0:
        return set()
    all_ids = set()
    offset = 0
    while offset < total:
        raw = store._collection.get(include=[], limit=1000, offset=offset)
        batch_ids = raw.get("ids", [])
        if not batch_ids:
            break
        all_ids.update(batch_ids)
        offset += len(batch_ids)
    return all_ids


def _print_summary(reader):
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


def _delete_collection():
    print("\n" + "═" * 60)
    print("  SCOUT AI — DELETE CHROMADB COLLECTION")
    print("═" * 60)
    embedder = Embedder()
    store = VectorStore(embedder=embedder)
    count = store.count
    if count == 0:
        print("  ℹ️  Collection is already empty.\n")
        return
    confirm = input(f"  ⚠️  Delete {count} documents? [yes/no]: ").strip().lower()
    if confirm not in ("yes", "y"):
        print("\n  Cancelled.\n")
        return
    store.nuke_and_recreate()
    print(f"\n  ✅ Collection deleted. Run load_data.py to reload.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
