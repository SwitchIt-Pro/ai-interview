"""
src/interview_state.py
----------------------
Interview session state management.

Tracks:
  - Rolling conversation window (last 5 turns) for LLM context
  - Score tracker per evaluation area (Qwen's scores)
  - OpenAI meta-evaluation results per turn
  - Asked question IDs (never trimmed, for deduplication)
  - Overall weighted score
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """One complete Q&A exchange in the interview."""

    turn_number: int
    timestamp: str
    question_id: str
    question_text: str
    candidate_response: str
    evaluation_area: str
    role: str
    experience_level: str

    # Qwen's assessment
    qwen_score: float = 0.0          # 0-10
    qwen_rationale: str = ""
    qwen_strengths: List[str] = field(default_factory=list)
    qwen_weaknesses: List[str] = field(default_factory=list)
    qwen_rephrased: bool = False      # Did Qwen rephrase the RAG question?
    qwen_select_reason: str = ""     # Why Qwen picked this question

    # OpenAI meta-evaluation (added after Qwen evaluates)
    openai_question_verdict: str = ""       # approved/acceptable/poor
    openai_question_quality: float = 0.0   # 0-10
    openai_score: float = 0.0              # OpenAI's independent score
    openai_scoring_verdict: str = ""       # accurate/over-scored/under-scored
    openai_score_delta: float = 0.0        # openai - qwen
    openai_assessment: str = ""


@dataclass
class EvaluationAreaTracker:
    """Live score tracker for one evaluation area."""

    area: str
    weight: float               # percentage weight (e.g. 25.0)
    questions_to_ask: int
    questions_asked: int = 0
    total_qwen_score: float = 0.0
    avg_qwen_score: float = 0.0
    weighted_score: float = 0.0  # (avg/10) * weight
    threshold_breached: bool = False
    min_score: Optional[float] = None
    non_negotiable: bool = False
    non_negotiable_alert: bool = False

    def update(self, qwen_score: float) -> None:
        self.questions_asked += 1
        self.total_qwen_score += qwen_score
        self.avg_qwen_score = round(self.total_qwen_score / self.questions_asked, 1)
        self.weighted_score = round((self.avg_qwen_score / 10) * self.weight, 2)

        if self.min_score is not None and self.avg_qwen_score < self.min_score:
            self.threshold_breached = True
            if self.non_negotiable:
                self.non_negotiable_alert = True

    @property
    def is_complete(self) -> bool:
        return self.questions_asked >= self.questions_to_ask

    @property
    def display(self) -> str:
        return f"{self.avg_qwen_score}/10 ({self.weighted_score:.2f}/{self.weight:.0f}%)"


@dataclass
class InterviewSession:
    """Complete interview session state."""

    session_id: str
    role: str
    experience_level: str
    started_at: str

    # Area score trackers
    area_tracker: Dict[str, EvaluationAreaTracker] = field(default_factory=dict)

    # Rolling context window (last N turns) — fed into Qwen as context.
    # Only contains Qwen's own Q+score data. No OpenAI content.
    conversation_history: List[ConversationTurn] = field(default_factory=list)

    # Permanent full log of ALL turns — never trimmed.
    # Used for reports, JSON export, and OpenAI meta-eval storage.
    full_turn_log: List[ConversationTurn] = field(default_factory=list)

    # All asked IDs (never trimmed, for deduplication)
    asked_question_ids: Set[str] = field(default_factory=set)

    # Metrics
    total_turns: int = 0
    overall_score: float = 0.0   # sum of weighted scores

    # Meta-evaluation stats (Qwen's performance as audited by OpenAI)
    meta_eval_alerts: List[str] = field(default_factory=list)
    qwen_scoring_issues: int = 0
    qwen_question_issues: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asked_question_ids"] = list(self.asked_question_ids)
        # Remove the rolling window from export — full_turn_log is the source of truth
        d.pop("conversation_history", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ──────────────────────────────────────────────────────────────────
# State Manager
# ──────────────────────────────────────────────────────────────────

class InterviewStateManager:
    """
    Manages interview session state.

    Usage
    -----
    mgr = InterviewStateManager()
    mgr.start_session(
        role="Sales Executive",
        experience_level="Mid-level",
        areas=[
            {"area": "Objection Handling", "weight": 25, "questions": 2},
            {"area": "Communication",       "weight": 20, "questions": 2},
        ],
    )
    mgr.record_turn(...)
    """

    def __init__(self, max_history: int = 5):
        self._max_history = max_history
        self._session: Optional[InterviewSession] = None

    def start_session(
        self,
        role: str,
        experience_level: str,
        areas: List[Dict],
    ) -> InterviewSession:
        """
        Initialize a new interview session.

        Parameters
        ----------
        areas : list of dicts with keys:
          area           : evaluation area name  (e.g. "Objection Handling")
          weight         : percentage weight      (e.g. 25.0)
          questions      : how many questions to ask for this area
          min_score      : optional minimum acceptable score (0-10)
          non_negotiable : bool — whether breach triggers alert
        """
        from uuid import uuid4
        session_id = f"scout_{uuid4().hex[:8]}"

        area_tracker = {}
        for a in areas:
            area_name = a["area"]
            area_tracker[area_name] = EvaluationAreaTracker(
                area=area_name,
                weight=float(a["weight"]),
                questions_to_ask=int(a.get("questions", 2)),
                min_score=a.get("min_score"),
                non_negotiable=bool(a.get("non_negotiable", False)),
            )

        self._session = InterviewSession(
            session_id=session_id,
            role=role,
            experience_level=experience_level,
            started_at=datetime.now().isoformat(),
            area_tracker=area_tracker,
        )

        logger.info(
            "Session %s started — Role: %s, Level: %s, Areas: %s",
            session_id, role, experience_level, list(area_tracker.keys()),
        )

        return self._session

    @property
    def session(self) -> InterviewSession:
        if self._session is None:
            raise RuntimeError("Session not started. Call start_session() first.")
        return self._session

    # ── Turn Recording ────────────────────────────────────────────

    def record_turn(
        self,
        question_id: str,
        question_text: str,
        candidate_response: str,
        evaluation_area: str,
        qwen_score: float,
        qwen_rationale: str,
        qwen_strengths: List[str],
        qwen_weaknesses: List[str],
        qwen_rephrased: bool = False,
        qwen_select_reason: str = "",
    ) -> ConversationTurn:
        """
        Record a Q&A turn, update area scores.
        OpenAI meta-evaluation results are added separately via update_meta_eval().
        """
        self.session.total_turns += 1

        turn = ConversationTurn(
            turn_number=self.session.total_turns,
            timestamp=datetime.now().isoformat(),
            question_id=question_id,
            question_text=question_text,
            candidate_response=candidate_response,
            evaluation_area=evaluation_area,
            role=self.session.role,
            experience_level=self.session.experience_level,
            qwen_score=qwen_score,
            qwen_rationale=qwen_rationale,
            qwen_strengths=qwen_strengths,
            qwen_weaknesses=qwen_weaknesses,
            qwen_rephrased=qwen_rephrased,
            qwen_select_reason=qwen_select_reason,
        )

        # ── Rolling context window (5 turns, Qwen reads this) ─────────
        # Only Qwen's Q+score data is kept here — no OpenAI content.
        self.session.conversation_history.append(turn)
        if len(self.session.conversation_history) > self._max_history:
            self.session.conversation_history.pop(0)

        # ── Permanent log (all turns, never trimmed, for reports) ─────
        self.session.full_turn_log.append(turn)

        # Permanent dedupe set
        self.session.asked_question_ids.add(question_id)

        # Update area tracker
        if evaluation_area in self.session.area_tracker:
            self.session.area_tracker[evaluation_area].update(qwen_score)
            self._recalculate_overall_score()

            tracker = self.session.area_tracker[evaluation_area]
            if tracker.non_negotiable_alert:
                alert = f"NON-NEGOTIABLE BREACH — {evaluation_area}: {tracker.avg_qwen_score}/10 < min {tracker.min_score}"
                if alert not in self.session.meta_eval_alerts:
                    self.session.meta_eval_alerts.append(alert)

        logger.debug(
            "Turn %d recorded — Area: %s, Qwen score: %.1f/10",
            self.session.total_turns, evaluation_area, qwen_score,
        )

        return turn

    def update_meta_eval(
        self,
        turn_number: int,
        question_verdict: str,
        question_quality: float,
        openai_score: float,
        scoring_verdict: str,
        score_delta: float,
        openai_assessment: str,
        alerts: List[str],
    ) -> None:
        """
        Attach OpenAI meta-evaluation results to a recorded turn.
        Stored on the turn in full_turn_log (permanent).
        The rolling conversation_history is NOT updated with OpenAI data —
        Qwen only sees its own Q+score context, never OpenAI feedback.
        """
        # Update in full_turn_log (permanent record)
        turn = next(
            (t for t in self.session.full_turn_log if t.turn_number == turn_number),
            None,
        )
        if turn:
            turn.openai_question_verdict = question_verdict
            turn.openai_question_quality = question_quality
            turn.openai_score = openai_score
            turn.openai_scoring_verdict = scoring_verdict
            turn.openai_score_delta = score_delta
            turn.openai_assessment = openai_assessment

        # Track Qwen's performance metrics (as evaluated by OpenAI)
        if question_verdict == "poor":
            self.session.qwen_question_issues += 1
        if not (-1.5 <= score_delta <= 1.5):
            self.session.qwen_scoring_issues += 1

        for alert in alerts:
            if alert not in self.session.meta_eval_alerts:
                self.session.meta_eval_alerts.append(alert)

    # ── Progress & Control ────────────────────────────────────────

    def is_complete(self) -> bool:
        """All evaluation areas have received their required number of questions."""
        return all(t.is_complete for t in self.session.area_tracker.values())

    def get_next_area(self) -> Optional[str]:
        """
        Return the next evaluation area that needs questions.
        Priority: highest-weight incomplete area first.
        """
        incomplete = [
            (area, t)
            for area, t in self.session.area_tracker.items()
            if not t.is_complete
        ]
        if not incomplete:
            return None
        incomplete.sort(key=lambda x: x[1].weight, reverse=True)
        return incomplete[0][0]

    def get_context_string(self) -> str:
        """
        Format recent conversation for Qwen's context injection.

        IMPORTANT: Only includes Qwen's own Q+score+rationale data.
        NO candidate response text (avoids bias from simulated answers).
        NO OpenAI meta-evaluation content (Qwen must not see its own audit).
        Rolling window: last 5 turns only.
        """
        if not self.session.conversation_history:
            return "No previous conversation — this is the first question."

        lines = [f"=== Recent Interview Context (last {len(self.session.conversation_history)} turns) ==="]
        for t in self.session.conversation_history:
            lines.append(f"\nTurn {t.turn_number} | Area: {t.evaluation_area}")
            lines.append(f"  Question : {t.question_text}")
            lines.append(f"  Qwen Score: {t.qwen_score}/10 — {t.qwen_rationale[:150]}")
        return "\n".join(lines)

    def _recalculate_overall_score(self) -> None:
        total = sum(
            t.weighted_score
            for t in self.session.area_tracker.values()
            if t.questions_asked > 0
        )
        self.session.overall_score = round(total, 2)

    # ── Display ───────────────────────────────────────────────────

    def get_live_scorecard(self) -> str:
        """Formatted scorecard for terminal display."""
        lines = ["", "=" * 65, "  LIVE SCORECARD (Qwen-7B Interview Score)", "=" * 65]
        for area, t in self.session.area_tracker.items():
            label = area[:36].ljust(38)
            if t.questions_asked > 0:
                alert = " 🚨" if t.non_negotiable_alert else ("⚠️" if t.threshold_breached else "")
                value = f"{t.display}{alert}"
            else:
                value = "Not yet evaluated"
            lines.append(f"  {label}: {value}")
        lines.append("-" * 65)
        lines.append(f"  {'Overall (Qwen)'.ljust(38)}: {self.session.overall_score:.2f}/100")
        if self.session.meta_eval_alerts:
            lines.append("")
            lines.append("  🔍 META-EVAL ALERTS (OpenAI):")
            for a in self.session.meta_eval_alerts:
                lines.append(f"     • {a}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.session.to_json())
        logger.info("Session saved to: %s", filepath)
