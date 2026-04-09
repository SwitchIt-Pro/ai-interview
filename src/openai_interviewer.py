"""
src/openai_interviewer.py
-----------------------
OpenAI acting as the AI Interviewer.

Responsibilities:
  1. Given RAG-retrieved question candidates from ChromaDB, the AI selects
     the BEST question for the current parameter and conversation context.
  2. The AI may lightly rephrase the question to make it flow naturally.
  3. The AI ALSO scores the candidate's response (0-10) with reasoning.
     This score is then sent to OpenAI meta-evaluator for audit.

Architecture summary:
  Excel → OpenAI Embedder → ChromaDB
  OpenAI (RAG read) → select/rephrase question → ask candidate
  Candidate answers
  OpenAI → score the answer
  OpenAI Meta-Evaluator → evaluate question quality + scoring accuracy
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional

from .config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class OpenAIClient:
    """
    Raw OpenAI chat client used by the AI Interviewer.

    Used by:
      - OpenAIInterviewer  (question selection, rephrasing, scoring)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env.\n"
                "Add: OPENAI_API_KEY=sk-proj-..."
            )
        from openai import OpenAI
        self._client      = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model       = model or config.OPENAI_INTERVIEWER_MODEL
        self._temperature = temperature if temperature is not None else config.OPENAI_INTERVIEWER_TEMPERATURE

        logger.info(
            "OpenAIClient ready — model: %s",
            self._model,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> tuple:
        """
        Send messages to OpenAI Chat Completions.

        Parameters
        ----------
        messages    : list of {role, content} dicts
        temperature : override default temperature

        Returns
        -------
        tuple: (response_text: str, elapsed_seconds: float)
        """
        from openai import RateLimitError, APIConnectionError, APIStatusError

        temp = temperature if temperature is not None else self._temperature
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                t_start = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                )
                elapsed = time.perf_counter() - t_start
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError(f"Empty response from OpenAI interviewer.")
                return content.strip(), elapsed

            except RateLimitError as e:
                last_exc = e
                delay = 2.0 * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI rate limit (attempt %d/%d). Retrying in %.0fs...",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except APIConnectionError as e:
                last_exc = e
                delay = 2.0 * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI connection failed (attempt %d/%d). Retrying in %.0fs...",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except APIStatusError as e:
                if e.status_code >= 500:
                    last_exc = e
                    time.sleep(2.0 * (2 ** (attempt - 1)))
                else:
                    raise RuntimeError(f"OpenAI API error: {e}")

        raise RuntimeError(
            f"OpenAI Interviewer failed after {_MAX_RETRIES} retries. Last error: {last_exc}\n"
            f"Check your OPENAI_API_KEY and network connection."
        )

    def prompt(self, user_text: str, system: Optional[str] = None, temperature: Optional[float] = None) -> tuple:
        """Single-turn convenience wrapper. Returns (response_text, elapsed_seconds)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return self.chat(messages, temperature=temperature)

    def check_connection(self) -> bool:
        """Verify OpenAI API is accessible."""
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            logger.info("OpenAI Interviewer OK — model '%s' ready", self._model)
            return True
        except Exception as e:
            raise RuntimeError(
                f"OpenAI Interviewer connection check failed: {e}\n"
                f"Check your OPENAI_API_KEY in .env."
            )


# ──────────────────────────────────────────────────────────────────
# AI Interviewer — Question Selection + Scoring
# ──────────────────────────────────────────────────────────────────

class OpenAIInterviewer:
    """
    OpenAI acts as the AI interviewer.

    Two tasks:
      1. select_question()  — RAG candidates → AI picks best + rephrases
      2. score_response()   — Candidate's answer → AI assigns score 0-10

    The scoring from step 2 is later fed to OpenAI Meta-Evaluator for audit.

    Usage
    -----
    interviewer = OpenAIInterviewer()

    selected = interviewer.select_question(
        candidates=[...],          # from ChromaDB
        role="Sales Executive",
        experience_level="Mid-level",
        evaluation_area="Objection Handling",
        conversation_context="...",
    )
    # → {question_text, question_id, rephrased, reason, rag_context}

    scoring = interviewer.score_response(
        question_text=selected["question_text"],
        candidate_response="I always try to...",
        evaluation_area="Objection Handling",
        what_ai_listens_for="...",
        strong_signal_example="...",
        weak_signal_example="...",
        role="Sales Executive",
        experience_level="Mid-level",
    )
    # → {raw_score, signal_strength, rationale, strengths, weaknesses}
    """

    def __init__(self, client: Optional[OpenAIClient] = None):
        self._client = client or OpenAIClient()

    # ── 1. Question Selection from RAG Candidates ─────────────────

    def select_question(
        self,
        candidates: List[Dict],
        role: str,
        experience_level: str,
        evaluation_area: str,
        conversation_context: str,
        asked_ids: Optional[set] = None,
    ) -> Dict:
        """
        Feed RAG candidates to OpenAI.
        AI picks the most relevant question and may lightly rephrase it.

        Parameters
        ----------
        candidates           : list of ChromaDB results (dicts with id, metadata, document)
        role                 : e.g. "Sales Executive"
        experience_level     : e.g. "Mid-level"
        evaluation_area      : what this question targets
        conversation_context : recent conversation turns (string)
        asked_ids            : set of already-asked question IDs (for dedup)

        Returns
        -------
        dict:
          question_text  : final question to ask
          question_id    : ChromaDB ID of the base question
          rephrased      : bool
          reason         : AI's reasoning
          rag_context    : {what_ai_listens_for, strong_signal_example, ...}
        """
        # Filter out already asked
        if asked_ids:
            candidates = [c for c in candidates if c.get("id") not in asked_ids]

        if not candidates:
            raise ValueError(
                f"No available questions for evaluation area: {evaluation_area}. "
                f"All retrieved questions may have already been asked."
            )

        candidates_text = self._format_candidates(candidates)

        prompt = f"""You are a warm, friendly, and professional AI interviewer conducting a {role} interview for a {experience_level} candidate.
Your tone is encouraging, conversational, and supportive — like a great interviewer who puts candidates at ease while still being thorough.

## Your current goal
Evaluate the candidate on: **{evaluation_area}**

## Recent Conversation Context
{conversation_context}

## Retrieved Question Candidates (from your question database)
{candidates_text}

## Instructions
1. Select the BEST question from the candidates above that:
   - Directly evaluates "{evaluation_area}"
   - Flows naturally and conversationally after the recent conversation
   - Is appropriate for a {experience_level} {role} candidate
   - Has NOT been covered recently

### Repetition Avoidance (IMPORTANT)
Before selecting, read the Recent Conversation Context carefully.
- SKIP any candidate question that covers the **same scenario, theme, or sub-topic** already explored in a previous question — even if the wording is different.
  Example: if you already asked about "handling objections in pricing", do NOT ask another question about pricing pushback or negotiation.
- PREFER candidates that explore a **fresh angle** of "{evaluation_area}" not yet covered.
- If ALL candidates overlap with previous questions, pick the one with the LEAST overlap and note it in the reason.

2. Rephrase the question so it sounds warm and natural. You MUST:
   - Start with a SHORT, friendly transition phrase. Choose ONE from this numbered list,
     picking based on the current turn number to ensure VARIETY across the interview.
     Do NOT use the same opener twice in a row:
       1. "Love that — now let me ask you about..."
       2. "Thanks for sharing that. Moving on, I'd love to hear..."
       3. "Great context! So tell me..."
       4. "Appreciate you walking me through that. Let's switch gears —"
       5. "That's helpful to know. I'm curious now —"
       6. "Perfect. On a related note..."
       7. "Good stuff. Let me ask you something a bit different —"
       8. "Got it, that makes sense. Next up —"
   - The opener should match the flow of the conversation naturally
   - Keep the core intent of the question intact
   - Sound like a real human interviewer, not a form or test
   - Avoid robotic or overly formal language

Return ONLY this JSON (no other text):
```json
{{
  "selected_index": <0-based integer index from candidates above>,
  "final_question": "<the question to ask, with a warm transition opener>",
  "rephrased": <true or false>,
  "reason": "<one sentence: why this question best serves the current evaluation goal>"
}}
```"""

        raw, elapsed = self._client.prompt(prompt)
        logger.debug("AI Interviewer select_question took %.2fs", elapsed)
        print(f"     ⏱  AI question selection: [{elapsed:.1f}s]")

        try:
            idx, question, rephrased, reason = self._parse_selection(raw, candidates)
        except Exception as e:
            logger.warning("AI selection parse failed (%s). Using top candidate.", e)
            idx = 0
            question = self._extract_question_text(candidates[0])
            rephrased = False
            reason = "Fallback to top RAG result"

        selected = candidates[idx]
        return {
            "question_text": question,
            "question_id": selected["id"],
            "rephrased": rephrased,
            "reason": reason,
            "rag_context": self._extract_rag_context(selected),
        }

    def _parse_selection(self, raw: str, candidates: List[Dict]):
        """Parse AI's JSON selection response."""
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        json_str = m.group(1) if m else raw.strip()
        data = json.loads(json_str)

        idx = int(data.get("selected_index", 0))
        idx = max(0, min(idx, len(candidates) - 1))
        question = data.get("final_question", "").strip()
        rephrased = bool(data.get("rephrased", False))
        reason = data.get("reason", "")

        if not question:
            raise ValueError("Empty final_question in AI response")

        return idx, question, rephrased, reason

    # ── 2. Response Scoring ───────────────────────────────────────

    def score_response(
        self,
        question_text: str,
        candidate_response: str,
        evaluation_area: str,
        what_ai_listens_for: str,
        strong_signal_example: str,
        weak_signal_example: str,
        role: str,
        experience_level: str,
        conversation_context: str = "",
    ) -> Dict:
        """
        OpenAI scores the candidate's response on a 0-10 scale.

        This score + reasoning is later compared against the meta-evaluator's
        independent score to assess calibration accuracy.

        Returns
        -------
        dict:
          raw_score       : float 0-10
          signal_strength : "strong" | "moderate" | "weak"
          rationale       : AI's scoring explanation
          strengths       : list[str]
          weaknesses      : list[str]
          needs_follow_up : bool
          follow_up_suggestion : str | None
        """
        prompt = f"""You are a friendly and expert interview evaluator for a {role} position.
Your job is to fairly and empathetically assess the candidate's response, like a great hiring manager would.

## Role & Level
Role: {role}
Experience Level: {experience_level}

## Evaluation Area
{evaluation_area}

## What to Listen For (from question database)
{what_ai_listens_for}

## Strong Response Example
{strong_signal_example}

## Weak Response Example
{weak_signal_example}

## Recent Conversation Context
{conversation_context}

## Question Asked
{question_text}

## Candidate's Response
{candidate_response}

## Task
Score the response on a scale of 0-10 for "{evaluation_area}".
Be fair and specific. Acknowledge genuine effort even in weak responses.
- 8-10: Strong — exceeds expectations with specific, concrete evidence
- 6-7: Good — meets expectations with reasonable evidence
- 4-5: Moderate — partial alignment, some relevant points
- 2-3: Weak — minimal relevant content, vague or generic
- 0-1: No relevant content or off-topic

### Repetition Penalty (IMPORTANT)
Compare this response against the Recent Conversation Context above.
- If the candidate reuses the **exact same specific example or story** they already gave in a previous answer → deduct **1.5 to 2 points** from the score.
- If the candidate recycles the **same generic idea or theme** (e.g. "I always communicate clearly") without adding new detail → deduct **0.5 to 1 point**.
- If the response is **completely fresh with new specifics** → no deduction.
If a penalty applies, you MUST mention it clearly in the rationale and add it as a weakness, e.g. "Candidate reused the same example from an earlier question — a fresh example would have strengthened this answer."

When writing strengths and weaknesses, use encouraging, constructive language.
For example: "Clearly articulated the STAR framework" or "Could benefit from more specific examples".

Return ONLY this JSON:
```json
{{
  "raw_score": <float 0.0-10.0, one decimal place>,
  "signal_strength": "strong|moderate|weak",
  "rationale": "<2-3 sentences explaining the score with specific, constructive references>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>"],
  "needs_follow_up": <true|false>,
  "follow_up_suggestion": "<a warm, natural follow-up question if needed, or null>"
}}
```"""

        raw, elapsed = self._client.prompt(prompt, temperature=0.3)
        logger.debug("AI score_response took %.2fs", elapsed)
        print(f"     ⏱  AI scoring: [{elapsed:.1f}s]")

        try:
            return self._parse_score(raw)
        except Exception as e:
            logger.error("AI scoring parse failed: %s", e)
            return {
                "raw_score": 5.0,
                "signal_strength": "moderate",
                "rationale": "Evaluation parse error — manual review recommended.",
                "strengths": ["Response was provided"],
                "weaknesses": ["Could not fully evaluate"],
                "needs_follow_up": False,
                "follow_up_suggestion": None,
            }

    def _parse_score(self, raw: str) -> Dict:
        """Parse AI's scoring JSON response."""
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        json_str = m.group(1) if m else raw.strip()
        data = json.loads(json_str)

        raw_score = float(data["raw_score"])
        raw_score = max(0.0, min(10.0, raw_score))

        return {
            "raw_score": raw_score,
            "signal_strength": data.get("signal_strength", "moderate"),
            "rationale": data.get("rationale", ""),
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "needs_follow_up": bool(data.get("needs_follow_up", False)),
            "follow_up_suggestion": data.get("follow_up_suggestion"),
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _format_candidates(self, candidates: List[Dict]) -> str:
        """Format ChromaDB candidates as numbered list for AI prompt."""
        lines = []
        for i, c in enumerate(candidates):
            meta = c.get("metadata", {})
            q_text = self._extract_question_text(c)
            q_type = meta.get("question_type", "")
            area = meta.get("evaluation_area", "")
            sim = c.get("similarity", 0)
            lines.append(
                f"[{i}] (similarity: {sim:.3f})\n"
                f"    Question: {q_text}\n"
                f"    Type: {q_type} | Area: {area}"
            )
        return "\n\n".join(lines)

    def _extract_question_text(self, question_data: Dict) -> str:
        """Extract question text from ChromaDB result dict."""
        meta = question_data.get("metadata", {})
        text = meta.get("question_text", "")
        if not text:
            doc = question_data.get("document", "")
            for line in doc.split("\n"):
                if line.startswith("Question:"):
                    text = line.replace("Question:", "").strip()
                    break
        return text or question_data.get("document", "")[:200]

    def _extract_rag_context(self, question_data: Dict) -> Dict[str, str]:
        """Extract RAG evaluation context from ChromaDB metadata."""
        meta = question_data.get("metadata", {})
        return {
            "what_ai_listens_for": meta.get("what_ai_listens_for", ""),
            "strong_signal_example": meta.get("strong_signal_example", ""),
            "weak_signal_example": meta.get("weak_signal_example", ""),
            "evaluation_area": meta.get("evaluation_area", ""),
            "skill_clusters_tested": meta.get("skill_clusters_tested", ""),
            "question_type": meta.get("question_type", ""),
            "difficulty_score": meta.get("difficulty_score", ""),
        }
