#!/usr/bin/env python3
"""
scripts/run_interview.py
------------------------
Main entry point for the Scout AI Interview Engine.

ARCHITECTURE:
  Questions DB  : Excel → OpenAI embeddings → ChromaDB
  Interviewer   : OpenAI GPT (reads from ChromaDB via RAG, asks questions, scores)
  Meta-Evaluator: OpenAI GPT (evaluates question quality + scoring accuracy)

QUICK START:
  # First, load your questions (ONCE):
  python scripts/load_data.py

  # Then run an interview:
  python scripts/run_interview.py

  # Custom role and parameters:
  python scripts/run_interview.py \\
      --role "Sales Executive" \\
      --level "Mid-level" \\
      --areas "Objection Handling:25:2:6:yes" "Communication:20:2:5:no" \\
      --no-meta-eval

AREA FORMAT:
  "AREA_NAME:WEIGHT:QUESTIONS:MIN_SCORE:NON_NEGOTIABLE"
  Examples:
    "Objection Handling:25:2:6:yes"  → 25% weight, 2 questions, min 6, non-negotiable
    "Communication Skills:20:2::no"  → 20% weight, 2 questions, no minimum
    "Pipeline Management:20:2"       → 20% weight, 2 questions (short form)

OUTPUT:
  - Live terminal diagnostics after each turn
  - JSON and text reports saved to reports/
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.interview_runner import InterviewRunner
from src.report_generator import ReportGenerator

# ── Default area configuration ────────────────────────────────────

DEFAULT_AREAS = [
    {"area": "Objection Handling",   "weight": 34, "questions": 1, "min_score": 6.0, "non_negotiable": True},
    {"area": "Communication Skills", "weight": 33, "questions": 1},
    {"area": "Pipeline Management",  "weight": 33, "questions": 1},
]


def parse_area_string(area_str: str) -> dict:
    """
    Parse CLI area format: "Objection Handling:25:2:6:yes"
    → {area, weight, questions, min_score, non_negotiable}
    """
    parts = area_str.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid area format '{area_str}'.\n"
            f"Expected: 'AREA_NAME:WEIGHT:QUESTIONS:MIN_SCORE:NON_NEGOTIABLE'\n"
            f"Example:  'Objection Handling:25:2:6:yes'"
        )

    area = parts[0].strip()
    weight = float(parts[1])
    questions = int(parts[2]) if len(parts) > 2 else 2
    min_score_str = parts[3].strip() if len(parts) > 3 else ""
    nn_str = parts[4].strip().lower() if len(parts) > 4 else "no"

    return {
        "area": area,
        "weight": weight,
        "questions": questions,
        "min_score": float(min_score_str) if min_score_str else None,
        "non_negotiable": nn_str in ("yes", "y", "true", "1"),
    }


def print_config(role: str, level: str, areas: list, use_meta_eval: bool, simulate: bool = False, persona: str = "") -> None:
    """Print session configuration before starting."""
    total_weight = sum(a["weight"] for a in areas)
    total_questions = sum(a.get("questions", 2) for a in areas)

    print("\n" + "═" * 65)
    print("  SESSION CONFIGURATION")
    print("═" * 65)
    print(f"  Role            : {role}")
    print(f"  Level           : {level}")
    print(f"  Total Questions : {total_questions}")
    print(f"  Meta-Evaluation : {'✓ Enabled' if use_meta_eval else '✗ Disabled (use --meta-eval to enable)'}")
    if simulate:
        print(f"  Candidate Mode  : 🎭 SIMULATED by OpenAI (persona: {persona})")
    else:
        print(f"  Candidate Mode  : 👤 INTERACTIVE (you type answers)")
    print(f"\n  {'Evaluation Area':<30} {'Weight':>6}  {'Qs':>3}  {'Min':>4}  {'NN':>3}")
    print("  " + "─" * 55)
    for a in areas:
        min_str = f"{a.get('min_score'):.0f}" if a.get("min_score") else "—"
        nn_str = "✓" if a.get("non_negotiable") else "—"
        print(
            f"  {a['area']:<30} {a['weight']:>5.0f}%  "
            f"{a.get('questions', 2):>3}  {min_str:>4}  {nn_str:>3}"
        )
    print("  " + "─" * 55)
    print(f"  {'TOTAL':<30} {total_weight:>5.0f}%")

    if abs(total_weight - 100.0) > 0.5:
        print(f"\n  ⚠️  WARNING: Weights sum to {total_weight:.0f}% (should be 100%)")

    print("═" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Scout AI Interviewer — OpenAI GPT + RAG + Meta-Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--role",
        default="Sales Executive",
        help="Job role being interviewed for (default: 'Sales Executive')"
    )
    parser.add_argument(
        "--level",
        default="Mid-level",
        choices=["Fresher", "Early Career", "Mid-level", "Senior"],
        help="Experience level (default: Mid-level)"
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        help=(
            "Evaluation areas in format AREA:WEIGHT:QUESTIONS:MIN_SCORE:NON_NEGOTIABLE\n"
            "Example: 'Objection Handling:25:2:6:yes' 'Communication:20:2::no'"
        ),
    )
    parser.add_argument(
        "--meta-eval",
        action="store_true",
        help="Enable OpenAI meta-evaluation after each turn (off by default)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode — you type the answers (default: OpenAI simulates)"
    )
    parser.add_argument(
        "--persona",
        default="average",
        choices=["strong", "average", "weak"],
        help="Candidate persona for simulation (default: average). Used with --simulate."
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save JSON and text reports to the reports/ directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # ── Build area configs ─────────────────────────────────────────
    if args.areas:
        try:
            areas = [parse_area_string(a) for a in args.areas]
        except ValueError as e:
            print(f"\n  ❌ Area configuration error: {e}")
            sys.exit(1)
    else:
        areas = DEFAULT_AREAS
        print("\n  ℹ️  No --areas specified. Using default Sales configuration.")

    use_meta_eval = args.meta_eval
    simulate      = not args.interactive
    persona       = args.persona

    # ── Print config ───────────────────────────────────────────────
    print_config(args.role, args.level, areas, use_meta_eval, simulate, persona)

    # ── Set up InterviewRunner ─────────────────────────────────────
    runner = InterviewRunner()

    try:
        runner.setup_session(
            role=args.role,
            experience_level=args.level,
            areas=areas,
            use_meta_eval=use_meta_eval,
            simulate=simulate,
            persona=persona,
        )
    except (RuntimeError, ValueError) as e:
        print(f"\n  ❌ Setup failed: {e}")
        sys.exit(1)

    # ── Run interactive interview ──────────────────────────────────
    try:
        runner.run_interactive()
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interview interrupted by user.")

    # ── Generate reports ───────────────────────────────────────────
    session = runner.get_session()
    generator = ReportGenerator()
    generator.print_summary(session)

    # ── OpenAI final review of interviewer performance ────────────
    if use_meta_eval:
        try:
            from src.openai_evaluator import OpenAIMetaEvaluator
            meta_evaluator = OpenAIMetaEvaluator()
            print("\n" + "═" * 65)
            print("  OPENAI REVIEW — Interviewer Performance")
            print("═" * 65)
            print("  ⏳ OpenAI is writing its final review of the interview...")
            review = meta_evaluator.review_qwen_performance(session)
            print()
            for line in review.splitlines():
                print(f"  {line}")
            print("═" * 65)
        except Exception as e:
            print(f"\n  ⚠️  Could not generate Qwen performance review: {e}")

    if args.export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path("reports/")
        export_dir.mkdir(parents=True, exist_ok=True)

        json_path = str(export_dir / f"session_{timestamp}.json")
        text_path = str(export_dir / f"report_{timestamp}.txt")

        generator.save_json(session, json_path)
        generator.save_text(session, text_path)
        print(f"  📁 Reports saved to {export_dir}")

    print("\n  ✅ Interview complete.\n")


if __name__ == "__main__":
    main()
