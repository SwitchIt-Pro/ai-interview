"""
scripts/inspect_db.py
----------------------
Inspect and verify ChromaDB contents.

Usage:
  python scripts/inspect_db.py              # Show collection stats
  python scripts/inspect_db.py --search "objection handling"
  python scripts/inspect_db.py --role "Sales Executive" --level "Mid-level"
  python scripts/inspect_db.py --id Q0001
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_store import VectorStore
from src.embedder import Embedder


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ChromaDB vector store",
    )
    parser.add_argument("--search", help="Semantic search query")
    parser.add_argument("--role", help="Filter by role")
    parser.add_argument("--level", help="Filter by experience level")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--id", help="Look up a specific question ID")
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  SCOUT AI — CHROMADB INSPECTOR")
    print("═" * 65)

    embedder = Embedder()
    store = VectorStore(embedder=embedder)

    print(f"\n  Collection : {store._collection_name}")
    print(f"  Documents  : {store.count}")

    if store.count == 0:
        print("\n  ⚠️  ChromaDB is empty. Run: python scripts/load_data.py")
        return

    if args.id:
        result = store.get_by_id(args.id)
        if result:
            print(f"\n  Question ID: {args.id}")
            print(f"  Document   : {result['document'][:300]}...")
            print(f"  Metadata   : ")
            for k, v in result["metadata"].items():
                if v:
                    print(f"    {k}: {v}")
        else:
            print(f"\n  ❌ Question '{args.id}' not found.")
        return

    if args.search:
        print(f"\n  Semantic Search: '{args.search}'")
        where = None
        if args.role:
            where = {"role": args.role}
        results = store.search(args.search, top_k=args.top_k, where=where)
        _print_results(results)
        return

    if args.role:
        print(f"\n  Filter: role='{args.role}', level='{args.level}'")
        conds = [{"role": args.role}]
        if args.level:
            conds.append({"experience_level": args.level})
        where = conds[0] if len(conds) == 1 else {"$and": conds}
        results = store.filter_by_metadata(where=where, limit=args.top_k)
        _print_results(results)
        return

    # Default: show sample
    print("\n  Sample questions (first 5):")
    results = store.filter_by_metadata(
        where={"status": "Active"},
        limit=5,
    )
    _print_results(results)


def _print_results(results):
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        sim = r.get("similarity", "")
        sim_str = f"  [sim: {sim:.3f}]" if sim else ""
        print(f"\n  {i}. [{r.get('id', '?')}]{sim_str}")
        print(f"     Q: {meta.get('question_text', r.get('document', '')[:100])}")
        print(f"     Role: {meta.get('role', '?')} | Level: {meta.get('experience_level', '?')} | Area: {meta.get('evaluation_area', '?')}")


if __name__ == "__main__":
    main()
