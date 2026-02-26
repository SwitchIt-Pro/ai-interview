"""
src/report_generator.py
------------------------
Generates interview reports after all turns are complete.

Three outputs:
  1. terminal_summary()  — condensed scorecard printed to console
  2. json_report()       — full structured JSON for storage/API
  3. text_report()       — human-readable recruiter report

Report includes:
  - Overall score (Qwen-7B based)
  - Per-area breakdown (score, signal, Qwen's evidence)
  - OpenAI meta-evaluation summary (question quality + scoring accuracy)
  - Red flags (non-negotiable breaches, poor questions, scoring issues)
  - Complete conversation transcript
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .interview_state import InterviewSession, InterviewStateManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates post-interview reports.

    Usage
    -----
    generator = ReportGenerator()
    generator.save_json(session, "reports/session.json")
    generator.save_text(session, "reports/report.txt")
    generator.print_summary(session)
    """

    def print_summary(self, session: InterviewSession) -> None:
        """Print a condensed summary to terminal."""

        print("\n" + "═" * 65)
        print("  SCOUT AI INTERVIEW — FINAL REPORT")
        print("═" * 65)
        print(f"  Session  : {session.session_id}")
        print(f"  Role     : {session.role}")
        print(f"  Level    : {session.experience_level}")
        print(f"  Duration : {session.total_turns} questions")
        print(f"  Date     : {session.started_at[:10]}")

        print("\n  EVALUATION AREA SCORES (Qwen-7B)")
        print("  " + "─" * 60)

        for area, t in session.area_tracker.items():
            label = area[:35].ljust(37)
            if t.questions_asked > 0:
                signal = (
                    "STRONG" if t.avg_qwen_score >= 7
                    else ("MODERATE" if t.avg_qwen_score >= 5 else "WEAK")
                )
                alert = " 🚨" if t.non_negotiable_alert else ("⚠️" if t.threshold_breached else "")
                print(f"  {label}: {t.avg_qwen_score:.1f}/10  [{signal}]{alert}")
                print(f"  {'':37}  Weighted: {t.weighted_score:.2f}/{t.weight:.0f}%")
            else:
                print(f"  {label}: Not evaluated")

        print("\n  " + "─" * 60)
        print(f"  {'OVERALL SCORE (Qwen)'.ljust(37)}: {session.overall_score:.2f}/100")

        # Meta-evaluation summary
        print(f"\n  OPENAI META-EVALUATION SUMMARY")
        print("  " + "─" * 60)
        print(f"  Questions audited        : {session.total_turns}")
        print(f"  Question quality issues  : {session.qwen_question_issues}")
        print(f"  Scoring accuracy issues  : {session.qwen_scoring_issues}")

        if session.meta_eval_alerts:
            print("\n  ⚠️  ALERTS:")
            for alert in session.meta_eval_alerts:
                print(f"     • {alert}")
        else:
            print("\n  ✅ No critical alerts from meta-evaluation")

        print("═" * 65 + "\n")

    def save_json(self, session: InterviewSession, filepath: str) -> None:
        """Save full session data as JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(session.to_json())
        print(f"  💾 JSON report saved: {filepath}")

    def save_text(self, session: InterviewSession, filepath: str) -> None:
        """Save human-readable recruiter report as text."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        report = self._build_text_report(session)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  💾 Text report saved: {filepath}")

    def _build_text_report(self, session: InterviewSession) -> str:
        """Build the full text recruiter report."""
        lines = [
            "═" * 65,
            "  SCOUT AI INTERVIEWER — RECRUITER REPORT",
            "═" * 65,
            f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"  Session ID : {session.session_id}",
            f"  Role       : {session.role}",
            f"  Level      : {session.experience_level}",
            f"  Questions  : {session.total_turns}",
            "",
            "═" * 65,
            "  SECTION 1: OVERALL SCORE",
            "═" * 65,
            f"  Overall Score (Qwen-7B): {session.overall_score:.2f}/100",
            "",
        ]

        # Area breakdown
        lines += [
            "═" * 65,
            "  SECTION 2: EVALUATION AREA BREAKDOWN",
            "═" * 65,
        ]

        for area, t in session.area_tracker.items():
            lines.append(f"\n  ── {area} (Weight: {t.weight:.0f}%) ──")
            if t.questions_asked > 0:
                signal = (
                    "STRONG" if t.avg_qwen_score >= 7
                    else ("MODERATE" if t.avg_qwen_score >= 5 else "WEAK")
                )
                lines.append(f"  Questions Asked : {t.questions_asked}/{t.questions_to_ask}")
                lines.append(f"  Qwen Score      : {t.avg_qwen_score:.1f}/10  [{signal}]")
                lines.append(f"  Weighted Score  : {t.weighted_score:.2f}/{t.weight:.0f}%")
                if t.threshold_breached and t.min_score:
                    lines.append(f"  ⚠️  THRESHOLD BREACH: {t.avg_qwen_score:.1f} < min {t.min_score}")
                if t.non_negotiable_alert:
                    lines.append(f"  🚨 NON-NEGOTIABLE ALERT: Score below required minimum!")
            else:
                lines.append("  Not evaluated in this session.")

        # OpenAI meta-evaluation summary
        lines += [
            "",
            "═" * 65,
            "  SECTION 3: OPENAI META-EVALUATION OF QWEN",
            "═" * 65,
            f"  This section shows how OpenAI audited Qwen-7B's performance.",
            f"  Qwen question issues  : {session.qwen_question_issues}",
            f"  Qwen scoring issues   : {session.qwen_scoring_issues}",
            "",
        ]

        if session.meta_eval_alerts:
            lines.append("  ALERTS:")
            for alert in session.meta_eval_alerts:
                lines.append(f"    • {alert}")
        else:
            lines.append("  ✅ Qwen performed within acceptable bounds on all turns.")

        # Transcript
        lines += [
            "",
            "═" * 65,
            "  SECTION 4: INTERVIEW TRANSCRIPT",
            "═" * 65,
        ]

        for turn in session.full_turn_log:
            lines.append(f"\n  Turn {turn.turn_number}  |  {turn.evaluation_area}")
            lines.append(f"  Q [{turn.question_id}]: {turn.question_text}")
            if turn.qwen_rephrased:
                lines.append(f"     (Rephrased by Qwen: {turn.qwen_select_reason})")
            lines.append(f"  A: {turn.candidate_response}")
            lines.append(f"  Qwen Score: {turn.qwen_score}/10  —  {turn.qwen_rationale}")
            if turn.openai_question_verdict:
                lines.append(
                    f"  OpenAI Question Audit: {turn.openai_question_verdict.upper()} "
                    f"(quality: {turn.openai_question_quality:.1f}/10)"
                )
            if turn.openai_scoring_verdict:
                lines.append(
                    f"  OpenAI Scoring Audit: {turn.openai_scoring_verdict.upper()} "
                    f"(OpenAI score: {turn.openai_score}/10, Δ={turn.openai_score_delta:+.1f})"
                )
            lines.append("  " + "─" * 60)

        lines += ["", "═" * 65, "  END OF REPORT", "═" * 65]
        return "\n".join(lines)
