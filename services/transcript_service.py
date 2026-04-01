"""
services/transcript_service.py

Saves interview transcript and Phase 3 reports to timestamped files in logs/.
  - interview_YYYYMMDD_HHMMSS.txt           → raw conversation transcript
  - interview_YYYYMMDD_HHMMSS_report_hr.txt → HR report (role fit %, decision, scores)
  - interview_YYYYMMDD_HHMMSS_report_candidate.txt → personalised feedback for candidate
"""

import os
import json
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)

# ── Bar-render helper ──────────────────────────────────────────────────────────

def _render_bar(pct: int, width: int = 10) -> str:
    filled = round(pct / 10)
    return "█" * filled + "░" * (width - filled)


class TranscriptService:
    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ts         = timestamp
        self.filepath    = os.path.join(config.LOG_DIR, f"interview_{timestamp}.txt")
        self._entries    = []
        logger.info(f"Transcript will be saved to: {self.filepath}")

    def add(self, speaker: str, text: str):
        """Record a single turn (AI or Candidate) with a timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {speaker}: {text}"
        self._entries.append(entry)
        logger.info(entry)

        if config.SAVE_TRANSCRIPT:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

    def save(self):
        """Finalise and rewrite the full conversation transcript file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"Interview Transcript — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n".join(self._entries))
        logger.info(f"✓ Transcript saved: {self.filepath}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Report Persistence
    # ═══════════════════════════════════════════════════════════════════════════

    def save_hr_report(self, report: dict):
        """
        Saves a formatted HR report showing role fit %, decision, parameter breakdown,
        red flags, and strongest signal.
        """
        path = os.path.join(config.LOG_DIR, f"interview_{self._ts}_report_hr.txt")

        scores    = report.get("scores", {})
        red_flags = report.get("red_flags", ["None detected"])
        decision  = report.get("decision", "Unknown")
        fit_pct   = report.get("role_fit_pct", 0.0)

        def _row(label: str, key: str) -> str:
            raw_score = scores.get(key, 0.0)
            pct       = round((raw_score / 10.0) * 100)
            bar       = _render_bar(pct)
            return f"  {label:<26} {bar}  {pct}%"

        lines = [
            "=" * 50,
            "  SWITCHIT-PRO — ROUND 1 INTERVIEW REPORT",
            "=" * 50,
            f"  Candidate  : {report.get('candidate_name', 'Candidate')}",
            f"  Role       : {report.get('role', '')}",
            f"  Date       : {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            "-" * 50,
            "",
            f"  ROLE FIT SCORE:  {fit_pct}%",
            f"  Decision:        {decision.upper()}",
            "",
            "-" * 50,
            "  PARAMETER BREAKDOWN",
            "-" * 50,
            _row("Conceptual Clarity",   "conceptual_clarity"),
            _row("Resume Authenticity",  "resume_authenticity"),
            _row("Role Relevance",       "role_relevance"),
            _row("Communication",        "communication"),
            _row("Problem Solving",      "problem_solving"),
            "",
            "-" * 50,
            "  RED FLAGS",
            "-" * 50,
        ]
        for flag in red_flags:
            lines.append(f"  - {flag}")

        lines += [
            "",
            "-" * 50,
            "  STRONGEST SIGNAL",
            "-" * 50,
            f"  {report.get('strongest_signal', '')}",
            "",
            "=" * 50,
        ]

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        # Also save machine-readable JSON alongside
        json_path = path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ HR report saved: {path}")

    def save_candidate_report(self, feedback_text: str, candidate: dict):
        """
        Saves the warm candidate feedback report.
        Never reveals scores, numbers, or pass/fail outcome.
        """
        path = os.path.join(
            config.LOG_DIR, f"interview_{self._ts}_report_candidate.txt"
        )
        role = candidate.get("role", "the role")

        header = "\n".join([
            "=" * 50,
            "  YOUR INTERVIEW FEEDBACK — SWITCHIT-PRO",
            "=" * 50,
            "  Hi Candidate,",
            "",
            "  Thank you for completing your Round 1 interview.",
            "  Here is personalised feedback to help you grow:",
            "",
            "-" * 50,
            "",
        ])

        footer = "\n".join([
            "",
            "-" * 50,
            "  Best of luck in your career journey.",
            "  — SwitchIt-Pro Hiring Team",
            "=" * 50,
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(feedback_text)
            f.write(footer)

        logger.info(f"✓ Candidate feedback report saved: {path}")
