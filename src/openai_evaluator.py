"""
src/openai_evaluator.py
-----------------------
OpenAI acts as the META-EVALUATOR — it evaluates Qwen's performance.

OpenAI does TWO things:
  1. QUESTION QUALITY CHECK:
     Was the question Qwen selected appropriate?
     - Right difficulty for the role/level?
     - Does it actually test the intended evaluation area?
     - Is it clear and professionally phrased?

  2. SCORING ACCURACY CHECK:
     Did Qwen score the response fairly?
     - Is Qwen's score consistent with the response quality?
     - Is the rationale sound and evidence-backed?
     - Flag over-scoring or under-scoring tendencies

This is the "OpenAI evaluating Qwen" layer described in the system design.
OpenAI does NOT conduct the interview — it only audits Qwen's work.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 2.0


# ──────────────────────────────────────────────────────────────────
# Result Dataclasses
# ──────────────────────────────────────────────────────────────────

@dataclass
class QuestionQualityResult:
    """OpenAI's verdict on a question Qwen selected."""

    question_text: str

    # Quality scores (each 0-10)
    relevance_score: float       # Does it test the intended evaluation area?
    difficulty_score: float      # Appropriate for the role/level?
    clarity_score: float         # Is the question clear and professional?
    overall_quality: float       # Weighted average

    # Feedback
    verdict: str                 # "approved" | "acceptable" | "poor"
    feedback: str                # 2-3 sentences explaining the verdict
    suggested_improvement: Optional[str]  # If poor, what would be better?

    def summary(self) -> str:
        return (
            f"Question Quality: {self.verdict.upper()} "
            f"(Overall: {self.overall_quality:.1f}/10 | "
            f"Relevance: {self.relevance_score:.1f} | "
            f"Difficulty: {self.difficulty_score:.1f} | "
            f"Clarity: {self.clarity_score:.1f})"
        )


@dataclass
class ScoringAccuracyResult:
    """OpenAI's verdict on how accurately Qwen scored a response."""

    qwen_score: float            # What Qwen gave
    openai_score: float          # What OpenAI thinks is correct

    # Verdict
    verdict: str                 # "accurate" | "over-scored" | "under-scored" | "inconsistent"
    score_delta: float           # openai_score - qwen_score
    is_accurate: bool            # |delta| <= 1.5 is considered accurate

    # Reasoning
    assessment: str              # 2-3 sentences on scoring quality
    rationale_quality: str       # "good" | "vague" | "unsupported"
    qwen_rationale_feedback: str # How well did Qwen explain the score?

    def summary(self) -> str:
        direction = "▲" if self.score_delta > 0 else ("▼" if self.score_delta < 0 else "=")
        return (
            f"Scoring Accuracy: {self.verdict.upper()} "
            f"(Qwen: {self.qwen_score}/10 | "
            f"OpenAI: {self.openai_score}/10 | "
            f"Delta: {direction}{abs(self.score_delta):.1f})"
        )


@dataclass
class MetaEvaluationResult:
    """Combined OpenAI meta-evaluation for one interview turn."""

    turn_number: int
    question_id: str
    question_quality: QuestionQualityResult
    scoring_accuracy: ScoringAccuracyResult

    # Final flags
    has_quality_issue: bool  # question was poor
    has_scoring_issue: bool  # scoring was inaccurate
    alerts: List[str]        # Non-negotiable issues to flag

    def print_diagnostics(self) -> None:
        """Print formatted diagnostics to terminal for inspection."""
        icon_q = "✅" if self.question_quality.verdict == "approved" else (
            "⚠️" if self.question_quality.verdict == "acceptable" else "❌"
        )
        icon_s = "✅" if self.scoring_accuracy.is_accurate else (
            "⚠️" if abs(self.scoring_accuracy.score_delta) <= 2.5 else "❌"
        )
        print(f"""
{'─' * 65}
  OPENAI META-EVALUATOR — Turn {self.turn_number}
{'─' * 65}
  {icon_q} {self.question_quality.summary()}
     Feedback: {self.question_quality.feedback}
     {('Suggestion: ' + self.question_quality.suggested_improvement) if self.question_quality.suggested_improvement else ''}

  {icon_s} {self.scoring_accuracy.summary()}
     Assessment: {self.scoring_accuracy.assessment}
     Rationale Quality: {self.scoring_accuracy.rationale_quality.upper()}
     Qwen Rationale Feedback: {self.scoring_accuracy.qwen_rationale_feedback}
{'─' * 65}""")


# ──────────────────────────────────────────────────────────────────
# OpenAI Meta-Evaluator
# ──────────────────────────────────────────────────────────────────

class OpenAIMetaEvaluator:
    """
    Uses OpenAI GPT-4o-mini to evaluate Qwen's interviewing performance.

    This is the only place OpenAI is used in the system.
    It does NOT conduct the interview — it audits Qwen's work.

    Usage
    -----
    evaluator = OpenAIMetaEvaluator()

    result = evaluator.evaluate_turn(
        turn_number=1,
        question_id="Q0042",
        question_text="How do you handle...",
        evaluation_area="Objection Handling",
        role="Sales Executive",
        experience_level="Mid-level",
        candidate_response="I typically...",
        qwen_score=7.0,
        qwen_rationale="Candidate demonstrated...",
        rag_context={...},
    )
    result.print_diagnostics()
    """

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env.\n"
                "Add: OPENAI_API_KEY=sk-proj-..."
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_EVAL_MODEL
        self._temperature = config.OPENAI_EVAL_TEMPERATURE
        logger.info("OpenAI meta-evaluator ready — model: %s", self._model)

    def evaluate_turn(
        self,
        turn_number: int,
        question_id: str,
        question_text: str,
        evaluation_area: str,
        role: str,
        experience_level: str,
        candidate_response: str,
        qwen_score: float,
        qwen_rationale: str,
        qwen_strengths: List[str],
        qwen_weaknesses: List[str],
        rag_context: Dict[str, str],
    ) -> MetaEvaluationResult:
        """
        Evaluate one complete interview turn:
          1. Was Qwen's question appropriate?
          2. Did Qwen score the response accurately?

        Parameters
        ----------
        turn_number        : interview turn number (for display)
        question_id        : ChromaDB question ID
        question_text      : what Qwen asked
        evaluation_area    : what this turn was testing
        role               : interviewee job role
        experience_level   : seniority level
        candidate_response : what the candidate said
        qwen_score         : Qwen's raw score (0-10)
        qwen_rationale     : Qwen's explanation of its score
        qwen_strengths     : list Qwen identified
        qwen_weaknesses    : list Qwen identified
        rag_context        : ChromaDB metadata for this question

        Returns
        -------
        MetaEvaluationResult
        """
        logger.info(
            "OpenAI meta-evaluation — Turn %d, Q: %s, Qwen score: %.1f",
            turn_number, question_id, qwen_score,
        )

        # ── Evaluate question quality ──────────────────────────────
        q_result = self._evaluate_question_quality(
            question_text=question_text,
            evaluation_area=evaluation_area,
            role=role,
            experience_level=experience_level,
            rag_context=rag_context,
        )

        # ── Evaluate scoring accuracy ──────────────────────────────
        s_result = self._evaluate_scoring_accuracy(
            question_text=question_text,
            candidate_response=candidate_response,
            evaluation_area=evaluation_area,
            role=role,
            experience_level=experience_level,
            qwen_score=qwen_score,
            qwen_rationale=qwen_rationale,
            qwen_strengths=qwen_strengths,
            qwen_weaknesses=qwen_weaknesses,
            rag_context=rag_context,
        )

        has_quality_issue = q_result.verdict == "poor"
        has_scoring_issue = not s_result.is_accurate

        alerts = []
        if has_quality_issue:
            alerts.append(f"POOR QUESTION: '{question_text[:60]}...'")
        if has_scoring_issue and abs(s_result.score_delta) > 2:
            alerts.append(
                f"SIGNIFICANT SCORING ERROR: Qwen={qwen_score} vs "
                f"OpenAI={s_result.openai_score} (Δ={s_result.score_delta:+.1f})"
            )

        return MetaEvaluationResult(
            turn_number=turn_number,
            question_id=question_id,
            question_quality=q_result,
            scoring_accuracy=s_result,
            has_quality_issue=has_quality_issue,
            has_scoring_issue=has_scoring_issue,
            alerts=alerts,
        )

    # ── Question Quality Evaluation ───────────────────────────────

    def _evaluate_question_quality(
        self,
        question_text: str,
        evaluation_area: str,
        role: str,
        experience_level: str,
        rag_context: Dict[str, str],
    ) -> QuestionQualityResult:
        """Evaluate whether Qwen selected an appropriate question."""

        what_listens = rag_context.get("what_ai_listens_for", "Not available")

        prompt = f"""You are auditing an AI interviewer (Qwen-7B) that selected a question for an interview.

## Interview Context
Role: {role}
Experience Level: {experience_level}
Intended Evaluation Area: {evaluation_area}

## What the Question Should Assess (from database)
{what_listens}

## Question Qwen Selected and Asked
"{question_text}"

## Your Task
Evaluate the quality of this question selection on THREE dimensions:
1. Relevance (0-10): Does it actually test "{evaluation_area}"?
2. Difficulty (0-10): Is it appropriately challenging for a {experience_level} {role}?
3. Clarity (0-10): Is it clear, professional, and unambiguous?

Verdict:
- "approved" if overall >= 7.0
- "acceptable" if overall 5.0-6.9
- "poor" if overall < 5.0

Return ONLY this JSON:
```json
{{
  "relevance_score": <float 0-10>,
  "difficulty_score": <float 0-10>,
  "clarity_score": <float 0-10>,
  "overall_quality": <float 0-10, weighted average>,
  "verdict": "approved|acceptable|poor",
  "feedback": "<2-3 sentences explaining your verdict>",
  "suggested_improvement": <"a better phrasing or null if acceptable">
}}
```"""

        raw, elapsed = self._call_openai(prompt)
        logger.debug("OpenAI question quality eval took %.2fs", elapsed)
        print(f"     ⏱  OpenAI question quality eval: [{elapsed:.1f}s]")
        try:
            data = self._parse_json(raw)
            return QuestionQualityResult(
                question_text=question_text,
                relevance_score=float(data["relevance_score"]),
                difficulty_score=float(data["difficulty_score"]),
                clarity_score=float(data["clarity_score"]),
                overall_quality=float(data["overall_quality"]),
                verdict=data["verdict"],
                feedback=data["feedback"],
                suggested_improvement=data.get("suggested_improvement"),
            )
        except Exception as e:
            logger.error("Question quality parse failed: %s", e)
            return QuestionQualityResult(
                question_text=question_text,
                relevance_score=7.0, difficulty_score=7.0, clarity_score=7.0,
                overall_quality=7.0, verdict="acceptable",
                feedback="Could not evaluate — parse error.",
                suggested_improvement=None,
            )

    # ── Scoring Accuracy Evaluation ───────────────────────────────

    def _evaluate_scoring_accuracy(
        self,
        question_text: str,
        candidate_response: str,
        evaluation_area: str,
        role: str,
        experience_level: str,
        qwen_score: float,
        qwen_rationale: str,
        qwen_strengths: List[str],
        qwen_weaknesses: List[str],
        rag_context: Dict[str, str],
    ) -> ScoringAccuracyResult:
        """Evaluate whether Qwen's score was accurate and fair."""

        what_listens = rag_context.get("what_ai_listens_for", "Not available")
        strong_ex = rag_context.get("strong_signal_example", "Not available")
        weak_ex = rag_context.get("weak_signal_example", "Not available")
        strengths_str = "\n  - ".join(qwen_strengths) if qwen_strengths else "None listed"
        weaknesses_str = "\n  - ".join(qwen_weaknesses) if qwen_weaknesses else "None listed"

        prompt = f"""You are auditing an AI interviewer (Qwen-7B) that scored a candidate's response.
Your job: verify whether Qwen's score is accurate and its reasoning is sound.

## Interview Context
Role: {role}
Experience Level: {experience_level}
Question: "{question_text}"
Evaluation Area: {evaluation_area}

## Scoring Criteria (from question database)
What to listen for: {what_listens}
Strong response example: {strong_ex}
Weak response example: {weak_ex}

## Candidate's Response
"{candidate_response}"

## Qwen's Assessment
Score: {qwen_score}/10
Rationale: {qwen_rationale}
Strengths identified:
  - {strengths_str}
Weaknesses identified:
  - {weaknesses_str}

## Your Task
1. What score would YOU give this response (0-10)? Be independent of Qwen's score.
2. Is Qwen's score accurate? Consider accurate if |Qwen - OpenAI| <= 1.5
3. Rate the quality of Qwen's rationale: "good" | "vague" | "unsupported"

Verdict:
- "accurate": |delta| <= 1.5
- "over-scored": Qwen scored too high (delta < -1.5)
- "under-scored": Qwen scored too low (delta > 1.5)
- "inconsistent": Qwen's rationale doesn't match its score

Return ONLY this JSON:
```json
{{
  "openai_score": <float 0-10, your independent score>,
  "verdict": "accurate|over-scored|under-scored|inconsistent",
  "assessment": "<2-3 sentences analyzing Qwen's scoring accuracy>",
  "rationale_quality": "good|vague|unsupported",
  "qwen_rationale_feedback": "<1-2 sentences on how well Qwen explained the score>"
}}
```"""

        raw, elapsed = self._call_openai(prompt)
        logger.debug("OpenAI scoring accuracy eval took %.2fs", elapsed)
        print(f"     ⏱  OpenAI scoring accuracy eval: [{elapsed:.1f}s]")
        try:
            data = self._parse_json(raw)
            openai_score = float(data["openai_score"])
            openai_score = max(0.0, min(10.0, openai_score))
            delta = round(openai_score - qwen_score, 2)
            return ScoringAccuracyResult(
                qwen_score=qwen_score,
                openai_score=openai_score,
                verdict=data["verdict"],
                score_delta=delta,
                is_accurate=abs(delta) <= 1.5,
                assessment=data["assessment"],
                rationale_quality=data["rationale_quality"],
                qwen_rationale_feedback=data["qwen_rationale_feedback"],
            )
        except Exception as e:
            logger.error("Scoring accuracy parse failed: %s", e)
            return ScoringAccuracyResult(
                qwen_score=qwen_score,
                openai_score=qwen_score,
                verdict="accurate",
                score_delta=0.0,
                is_accurate=True,
                assessment="Could not evaluate — parse error.",
                rationale_quality="vague",
                qwen_rationale_feedback="Evaluation error.",
            )

    # ── End-of-Session Qwen Performance Review ────────────────────

    def review_qwen_performance(
        self,
        session,  # InterviewSession
    ) -> str:
        """
        After the full interview, ask OpenAI to write an overall
        performance review of Qwen-7B as the interviewer.

        Covers:
          - Question quality across all turns
          - Scoring calibration (was Qwen consistent and accurate?)
          - Bias or patterns noticed (e.g. over-generous, repetitive questions)
          - A final recommendation: Trustworthy / Needs Calibration / Unreliable

        Returns
        -------
        str: formatted multi-line review for terminal display
        """
        turns = session.full_turn_log
        if not turns:
            return "  No turns to review."

        # Build a concise turn-by-turn summary for OpenAI
        turn_summaries = []
        for t in turns:
            delta_dir = "▲" if t.openai_score_delta > 0 else ("▼" if t.openai_score_delta < 0 else "=")
            turn_summaries.append(
                f"Turn {t.turn_number} | Area: {t.evaluation_area}\n"
                f"  Question: {t.question_text[:120]}\n"
                f"  Qwen Score: {t.qwen_score}/10 | OpenAI Score: {t.openai_score}/10 | "
                f"Delta: {delta_dir}{abs(t.openai_score_delta):.1f}\n"
                f"  Question Verdict: {t.openai_question_verdict or 'N/A'} "
                f"(Quality: {t.openai_question_quality or 'N/A'}/10)\n"
                f"  Scoring Verdict: {t.openai_scoring_verdict or 'N/A'}"
            )

        turns_text = "\n\n".join(turn_summaries)

        avg_delta = (
            sum(t.openai_score_delta for t in turns) / len(turns)
            if turns else 0.0
        )
        scoring_issues  = session.qwen_scoring_issues
        question_issues = session.qwen_question_issues

        prompt = f"""You are writing a performance review of an AI interviewer called Qwen-7B.
Qwen conducted a {session.role} interview (Level: {session.experience_level}).
You have already evaluated each of Qwen's turns individually. Now write a final overall review.

## Interview Summary
- Total turns: {len(turns)}
- Question quality issues: {question_issues}/{len(turns)}
- Scoring accuracy issues: {scoring_issues}/{len(turns)}
- Average score delta (OpenAI - Qwen): {avg_delta:+.2f}
  (Positive = Qwen under-scored, Negative = Qwen over-scored)

## Turn-by-Turn Breakdown
{turns_text}

## Your Task
Write a structured performance review of Qwen as an interviewer. Cover:

1. **Question Quality** — Were Qwen's questions well-chosen, relevant, and appropriately challenging?
2. **Scoring Calibration** — Was Qwen's scoring consistent and fair across turns? Any bias (e.g. always harsh or always generous)?
3. **Notable Patterns** — Any recurring strengths or weaknesses in how Qwen performed?
4. **Overall Recommendation** — One of:
   - ✅ TRUSTWORTHY — Qwen performed reliably. Scores and questions can be trusted.
   - ⚠️ NEEDS CALIBRATION — Qwen showed some issues. Results should be reviewed carefully.
   - ❌ UNRELIABLE — Significant problems. Manual review strongly recommended.

Write in a professional but direct tone. Be specific — reference actual turn data when making a point.
Keep the total response to 200-300 words."""

        try:
            raw, elapsed = self._call_openai(prompt)
            logger.debug("OpenAI Qwen performance review took %.2fs", elapsed)
            print(f"     ⏱  OpenAI Qwen performance review: [{elapsed:.1f}s]")
            return raw.strip()
        except Exception as e:
            logger.error("Qwen performance review failed: %s", e)
            return f"  Could not generate Qwen performance review: {e}"

    # ── OpenAI API Call ───────────────────────────────────────────

    def _call_openai(self, prompt: str) -> tuple:
        """Call OpenAI with retry logic. Returns (content, elapsed_seconds)."""
        from openai import RateLimitError, APIConnectionError, APIStatusError

        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                t_start = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                )
                elapsed = time.perf_counter() - t_start
                return response.choices[0].message.content.strip(), elapsed

            except RateLimitError as e:
                last_exc = e
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI rate limit (attempt %d/%d). Retrying in %.0fs...",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except APIConnectionError as e:
                last_exc = e
                time.sleep(_BASE_DELAY)

            except APIStatusError as e:
                if e.status_code >= 500:
                    last_exc = e
                    time.sleep(_BASE_DELAY * (2 ** (attempt - 1)))
                else:
                    raise

        raise RuntimeError(
            f"OpenAI failed after {_MAX_RETRIES} retries. Last: {last_exc}"
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from markdown code block or raw string."""
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        json_str = m.group(1) if m else raw.strip()
        return json.loads(json_str)
