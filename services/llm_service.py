"""
services/llm_service.py

Phase 1 — Pre-Interview: Dynamic Q1/Q2 generation from resume + JD
Phase 2 — Live Interview: Per-answer scoring + n-2 adaptive question generation
Phase 3 — Post-Interview: HR report + candidate feedback generation

5 Scoring Parameters (each 0-10, total /50 = role fit %):
  1. Conceptual Clarity
  2. Resume Authenticity
  3. Role Relevance
  4. Communication & Clarity
  5. Problem-Solving Approach
"""

import sys
import json
import logging
import re
import time
from statistics import mean
from typing import Generator, Tuple, Optional, List, Dict, Any
from pathlib import Path
import random

import config

from .ollama_client import OllamaClient
from .rag_service import RAGService

logger = logging.getLogger(__name__)

# ── Question → Parameter mapping (handoff spec) ─────────────────────────────
QUESTION_PARAMETER = {
    1: "Communication",          # Q1 baseline intro
    2: "Communication",          # Q2 motivation
    3: "Resume Authenticity",    # Q3 generated from Q1 answer
    4: "Conceptual Clarity",     # Q4 generated from Q2 answer
    5: "Role Relevance",         # Q5 generated from Q3 answer (RAG adapted)
    6: "Problem Solving",        # Q6 generated from Q4 answer
    7: "Communication",          # Q7 closing
}

# Q(n) answer → generates Q(n+2) targeting this parameter
GENERATE_TARGET = {
    1: "Resume Authenticity",    # Q1 → Q3
    2: "Conceptual Clarity",     # Q2 → Q4
    3: "Role Relevance",         # Q3 → Q5
    4: "Problem Solving",        # Q4 → Q6
    5: "Communication",          # Q5 → Q7 (explain X to non-technical person)
}

# Fallback questions when VectorDB / Ollama is unavailable
FALLBACK_QUESTIONS = [
    "Could you walk me through a technical challenge you faced recently and how you resolved it?",
    "How do you approach learning a new technology or framework on the job?",
    "Can you describe how you would debug a slow API endpoint step by step?",
    "Tell me about a project where you had to make an important architectural decision.",
    "How do you ensure code quality in your day-to-day development process?",
]


def _extract_json(raw: str) -> dict:
    """Safely extract the first JSON object from an LLM response."""
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        return json.loads(raw[start:end])
    except Exception as e:
        logger.warning(f"JSON extraction failed: {e} | raw[:200]={raw[:200]}")
        return {}


class LLMService:
    def __init__(self):
        logger.info("Initializing LLMService (using native OllamaClient and RAGService)...")
        self.qwen = OllamaClient()
        self.rag = RAGService()

        self.meta_evaluator = None

        # session_id → session dict
        self.sessions: Dict[str, dict] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — PRE-INTERVIEW PREPARATION
    # ═══════════════════════════════════════════════════════════════════════════

    def prepare_session(
        self,
        session_id: str,
        role: str,
        level: str,
        industry: str,
        resume_text: str,
        jd_text: str,
    ):
        """
        Phase 1: Parse resume + JD, generate personalised Q1 & Q2, set up session object.
        Runs in a background thread — does NOT block /api/apply response.
        """
        logger.info(f"[{session_id}] Phase 1 preparation started...")

        # ── Retrieve VectorDB candidates (RAG Service) ───────────────────────
        try:
            candidates = self.rag.fetch_interview_questions(
                role=role, experience_level=level, n=10, industry=industry
            )
        except Exception as e:
            logger.warning(f"[{session_id}] VectorDB search failed (Ollama down?): {e}")
            candidates = []

        # ── Generate personalised Q1 + Q2 via Qwen ──────────────────────────
        phase1_prompt = f"""You are preparing the opening two questions for a 10-minute screening interview.

Resume: {resume_text}
Job Description: {jd_text}

Extract and return JSON only:
{{
  "is_eligible": true,
  "ineligibility_reason": "If is_eligible is false (meaning experience level/domain do NOT match JD strictly), provide a one sentence reason why.",
  "last_company": "most recent employer name from resume",
  "resume_personal_detail": "one specific detail from resume -- e.g. current role, a project they led, years of experience, or a skill they highlight",
  "jd_company_detail": "one specific detail about the company from JD -- e.g. their product domain, tech stack, mission statement, or growth stage",
  "q1": "personalised intro question using resume_personal_detail -- e.g. 'I see you have been working as a React developer at Infosys for 3 years -- could you walk me through your journey and what brought you to apply here?'",
  "q2": "personalised motivation question using jd_company_detail -- e.g. 'SwitchIt-Pro is building an AI-powered hiring platform -- what specifically drew you to this space and why did you choose us?'"
}}

Rules for Q1:
- Reference exactly one detail from their resume -- name, role, company, or project
- Ask them to walk through their background naturally
- One sentence, conversational, under 30 words

Rules for Q2:
- Reference exactly one specific thing about the company from the JD
- Do not ask a generic "why do you want to work here" question
- One sentence, conversational, under 30 words
- Must end with a question mark"""

        extracted = {}
        try:
            raw, _ = self.qwen.prompt(phase1_prompt)
            extracted = _extract_json(raw)
        except Exception as e:
            logger.error(f"[{session_id}] Phase 1 Q-gen failed: {e}")

        q1 = extracted.get("q1") or "Could you tell me a little bit about yourself and your background?"
        q2 = extracted.get("q2") or "What specifically drew you to apply to this role with us?"
        last_company = extracted.get("last_company", "")
        is_eligible = extracted.get("is_eligible", True)
        ineligibility_reason = extracted.get("ineligibility_reason", "")

        candidate_obj = {
            "role": role,
            "resume_summary": resume_text[:300] if resume_text else "",
            "jd_summary": jd_text[:300] if jd_text else "",
        }

        self.sessions[session_id] = {
            # ── Identity ──
            "candidate":      candidate_obj,
            "role":           role,
            "level":          level,
            "industry":       industry,
            "resume":         resume_text,
            "jd":             jd_text,
            "last_company":   last_company,
            "is_eligible":    is_eligible,
            "ineligibility_reason": ineligibility_reason,
            # ── Questions (starts with Q1+Q2, grows to 7 during Phase 2) ──
            "questions":      [q1, q2],
            # ── Phase 2 state ──
            "answers":        [],
            "score_objects":  [],   # full scoring dicts per answer
            "scores":         [],   # float scores only (for quick math)
            "timings":        [],   # response delay per answer (seconds)
            "followup_used":  {n: False for n in range(1, 8)},
            "asked_ids":      set(),
            "questions_queue": candidates,
            "turn":           0,
            "last_question":  None,
            "last_rag_context": None,
            "history":        f"Candidate is applying for {level} {role} in {industry}.",
            # ── Phase 3 ──
            "report":         None,
            "feedback":       None,
        }

        logger.info(
            f"[{session_id}] ✓ Phase 1 complete. Q1='{q1[:60]}' Q2='{q2[:60]}'"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — LIVE INTERVIEW
    # ═══════════════════════════════════════════════════════════════════════════

    def get_opening_line(self, session_id: str) -> str:
        """Returns the greeting that leads into Q1."""
        sess = self.sessions.get(session_id)
        if not sess:
            return "Hello. Let's begin your interview."
        q1 = sess["questions"][0]
        greeting = (
            f"Hello! Welcome to your SwitchIt-Pro Round 1 interview. "
            f"I'm your AI interviewer — this will take about 10 minutes. "
            f"{q1}"
        )
        sess["last_question"] = q1
        sess["history"] += f"\nInterviewer: {greeting}"
        sess["turn"] = 1
        return greeting

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _score_answer(
        self,
        n: int,
        question_text: str,
        answer_text: str,
        sess: dict,
    ) -> dict:
        """
        Calls Qwen to score a single answer. Returns full scoring dict.
        Falls back to a neutral score if LLM is unavailable.
        """
        parameter = QUESTION_PARAMETER.get(n, "Communication")
        role       = sess["role"]
        jd_summary = sess["candidate"]["jd_summary"]
        resume_sum = sess["candidate"]["resume_summary"]

        parameter_instruction = {
            "Communication":       "How clearly, confidently, and concisely did the candidate communicate? Penalise rambling, filler phrases, or off-topic answers.",
            "Resume Authenticity": "Does the answer show genuine depth about claimed experience? Red flag: vague or cannot explain their own project.",
            "Conceptual Clarity":  "Can they explain the 'why' not just the 'what'? Red flag: buzzwords with no explanation.",
            "Role Relevance":      "Is the answer tied specifically to this role and JD? Red flag: generic answers not connected to the role.",
            "Problem Solving":     "Does the candidate reason step-by-step? Red flag: jumps to answer without visible reasoning process.",
        }.get(parameter, "Assess the overall quality of the answer.")

        prompt = f"""You are a hiring evaluator scoring a single interview answer.

Role: {role}
Job Description summary: {jd_summary}
Resume summary: {resume_sum}
Question (Q{n}): {question_text}
Candidate's answer: {answer_text}

Score on the parameter below (0-10). Be strict. Reserve 9-10 for exceptional only.
Parameter: {parameter_instruction}

Check if this answer contradicts anything on the candidate's resume.

Return JSON only:
{{
  "parameter": "{parameter}",
  "score": <0-10>,
  "resume_contradiction": <true or false>,
  "contradiction_detail": "<one line or null>",
  "answer_quality": "<strong | acceptable | weak>",
  "key_signal": "<one phrase — most notable thing about this answer>",
  "depth": "<surface | moderate | deep>",
  "acknowledgement": "<one short conversational sentence (under 15 words) warmly acknowledging their answer, without asking a question>",
  "is_abusive": <true or false>
}}

Note: set is_abusive to true ONLY if the candidate uses abusive, disrespectful, or extremely inappropriate language."""

        try:
            raw, _ = self.qwen.prompt(prompt)
            result = _extract_json(raw)
            # Validate types
            result["score"] = float(result.get("score", 5.0))
            result.setdefault("answer_quality", "acceptable")
            result.setdefault("depth", "moderate")
            result.setdefault("key_signal", "")
            result.setdefault("resume_contradiction", False)
            result.setdefault("contradiction_detail", None)
            result.setdefault("acknowledgement", "Got it, thanks for sharing that.")
            result.setdefault("is_abusive", False)
            return result
        except Exception as e:
            logger.error(f"Scoring Q{n} failed: {e}")
            return {
                "parameter": parameter,
                "score": 5.0,
                "resume_contradiction": False,
                "contradiction_detail": None,
                "answer_quality": "acceptable",
                "key_signal": "",
                "depth": "moderate",
                "acknowledgement": "Got it, thanks for sharing that.",
                "is_abusive": False,
            }

    # ── Next-Question Generation ─────────────────────────────────────────────

    def _generate_next_question(
        self,
        n: int,
        answer_text: str,
        score_obj: dict,
        sess: dict,
    ) -> str:
        """
        n-2 adaptive: scoring Q(n) → generates Q(n+2).
        Uses the score signal to probe weakness or advance difficulty.
        """
        target_param = GENERATE_TARGET.get(n)
        if not target_param:
            return random.choice(FALLBACK_QUESTIONS)

        role         = sess["role"]
        jd_summary   = sess["candidate"]["jd_summary"]
        resume_sum   = sess["candidate"]["resume_summary"]
        history      = sess["history"]
        answer_quality = score_obj.get("answer_quality", "acceptable")
        score        = score_obj.get("score", 5.0)
        depth        = score_obj.get("depth", "moderate")
        key_signal   = score_obj.get("key_signal", "")
        contradiction = score_obj.get("resume_contradiction", False)

        # For Q5 (Role Relevance), try to adapt a VectorDB question to the context
        rag_hint = ""
        if n == 3:  # generating Q5
            try:
                candidates = sess.get("questions_queue", [])
                if candidates:
                    for cand in candidates:
                        if cand not in sess["asked_ids"]:
                            sess["asked_ids"].add(cand)
                            rag_hint = f"\nBase question from question bank (adapt it to depth level): {cand}"
                            sess["last_rag_context"] = {}
                            break
            except Exception:
                pass

        prompt = f"""You are an interviewer generating the next question in a live screening interview.

Role: {role}
Job Description summary: {jd_summary}
Resume summary: {resume_sum}
Interview history so far: {history}

You just scored Q{n}:
  Answer quality: {answer_quality}
  Score: {score}/10
  Depth: {depth}
  Key signal: {key_signal}
  Resume contradiction: {contradiction}{rag_hint}

Generate Q{n+2}. Rules:
- If answer_quality is "weak" or score < 5: probe deeper into the weakness or contradiction.
- If answer_quality is "strong" and score >= 8: advance to a harder or more specific angle.
- If answer_quality is "acceptable": build naturally on what they said.
- If depth is "surface": ask for a concrete example or implementation detail.
- Target parameter for Q{n+2}: {target_param}
- Do not reference scores or evaluation to the candidate.
- One sentence. Conversational. Less than 30 words.
- NO PREAMBLE. NO EXPLANATION.

Return exactly this JSON format and nothing else:
{{
  "thought_process": "Briefly plan the angle of your question behind the scenes.",
  "question": "The actual question you will say to the candidate."
}}"""

        try:
            raw, _ = self.qwen.prompt(prompt)
            result = _extract_json(raw)
            q = result.get("question", "")
            if not q:
                logger.warning("No question extracted from JSON. raw=" + raw[:100])
                fallback_idx = (n - 1) % len(FALLBACK_QUESTIONS)
                return FALLBACK_QUESTIONS[fallback_idx]
            
            # Additional cleanup just in case it leaks "Question:" or "Generated Q:"
            q = re.sub(r'^(?:\*\*.*?\*\*|Generated Q\d?:.*?|Question:.*?|Here is.*?:\s*)', '', q, flags=re.IGNORECASE).strip()

            if len(q) > 10:
                return q
        except Exception as e:
            logger.error(f"generate_next_question failed for n={n}: {e}")

        fallback_idx = (n - 1) % len(FALLBACK_QUESTIONS)
        return FALLBACK_QUESTIONS[fallback_idx]

    # ── Main streaming loop ──────────────────────────────────────────────────

    def generate_stream(
        self, session_id: str, user_text: str, timing: float = 0.0
    ) -> Generator[Tuple[str, Optional[float]], None, None]:
        """
        Called after each candidate answer.
        1. Records the answer + timing
        2. Scores Q(turn-1)
        3. In background, generates Q(turn+1) using n-2 signal
        4. Yields the next question word-by-word
        """
        sess = self.sessions.get(session_id)
        if not sess:
            yield ("Session not found.", None)
            return

        turn = sess.get("turn", 0)
        logger.info(f"[{session_id}] generate_stream: turn={turn}")

        # ── Record answer & timing ───────────────────────────────────────────
        sess["answers"].append(user_text)
        sess["timings"].append(timing)
        sess["history"] += f"\nCandidate: {user_text}"

        # ── Closing: turn >= 7 ───────────────────────────────────────────────
        if turn >= 7:
            closing = (
                "Thank you so much for your time today — that wraps up my questions. "
                "Someone from the SwitchIt-Pro recruitment team will be in touch with you very shortly. "
                "It was great speaking with you — have a wonderful day!"
            )
            for w in closing.split():
                yield (w + " ", None)
            # Trigger Phase 3 in background after closing
            import threading
            threading.Thread(
                target=self._run_phase3, args=(session_id,), daemon=True
            ).start()
            return

        # ── Score the answer for the current turn ────────────────────────────
        qwen_score = None
        score_obj  = None
        if sess.get("last_question"):
            # Yield a keepalive sentinel BEFORE the blocking Ollama scoring call.
            # The server thread picks this up and sends a status ping to the WebSocket
            # so the browser doesn't see a silent connection and drop it.
            yield (None, None)   # sentinel: "I'm alive, scoring now"
            logger.info(f"[{session_id}] Scoring Q{turn}...")
            score_obj  = self._score_answer(turn, sess["last_question"], user_text, sess)
            qwen_score = score_obj["score"]
            sess["score_objects"].append(score_obj)
            sess["scores"].append(qwen_score)
            logger.info(f"[{session_id}] Q{turn} → {qwen_score}/10 ({score_obj.get('answer_quality')})")

            # ── Detect Abuse & Ban ───────────────────────────────────────────
            if score_obj.get("is_abusive", False):
                ban_msg = "You have been disrespectful. Your interview is terminated, and you are banned for 15 days from SwitchIt."
                sess["history"] += f"\nInterviewer: {ban_msg} [USER BANNED]"
                for word in ban_msg.split():
                    yield (word + " ", None)
                yield ("__BAN__", None)
                return

            # ── n-2 adaptive: generate Q(turn+2) using this score ────────────
            if score_obj and (turn + 2) <= 7:
                import threading
                def _gen_q():
                    try:
                        next_q = self._generate_next_question(turn, user_text, score_obj, sess)
                        # Append only if we don't already have it
                        if len(sess["questions"]) < turn + 2:
                            sess["questions"].append(next_q)
                            logger.info(
                                f"[{session_id}] Q{turn+2} generated (adaptive): '{next_q[:60]}'"
                            )
                    except Exception as e:
                        logger.error(f"[{session_id}] Adaptive Q gen failed: {e}")
                threading.Thread(target=_gen_q, daemon=True).start()

        # ── Pick the next question to ask ────────────────────────────────────
        next_q       = ""
        rag_context  = {}

        try:
            if turn < len(sess["questions"]):
                # Already generated (Phase 1 Q1/Q2 or n-2 adaptive)
                next_q = sess["questions"][turn]
            else:
                # Fallback: pick from VectorDB if question not yet generated
                logger.info(f"[{session_id}] Q{turn+1} not ready yet — using VectorDB fallback")
                candidates = sess.get("questions_queue", [])
                if candidates:
                    selected_q = None
                    for cand in candidates:
                        if cand not in sess["asked_ids"]:
                            sess["asked_ids"].add(cand)
                            selected_q = cand
                            break
                            
                    next_q = selected_q if selected_q else FALLBACK_QUESTIONS[turn % len(FALLBACK_QUESTIONS)]
                    rag_context = {}
                    # Store it so n-2 logic can reference it
                    while len(sess["questions"]) <= turn:
                        sess["questions"].append(next_q)
                else:
                    next_q = FALLBACK_QUESTIONS[turn % len(FALLBACK_QUESTIONS)]
                    while len(sess["questions"]) <= turn:
                        sess["questions"].append(next_q)

        except Exception as e:
            logger.error(f"[{session_id}] Question selection failed: {e}")
            next_q = FALLBACK_QUESTIONS[turn % len(FALLBACK_QUESTIONS)]

        # ── Advance state ────────────────────────────────────────────────────
        sess["turn"]             = turn + 1
        sess["last_question"]    = next_q
        sess["last_rag_context"] = rag_context
        
        full_reply = next_q
        if score_obj and "acknowledgement" in score_obj:
            full_reply = score_obj["acknowledgement"] + " " + next_q

        sess["history"]         += f"\nInterviewer: {full_reply}"

        # ── Yield next question word-by-word ─────────────────────────────────
        for word in full_reply.split():
            yield (word + " ", qwen_score)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — HIDDEN SIGNAL CAPTURE
    # ═══════════════════════════════════════════════════════════════════════════

    def _capture_hidden_signals(self, sess: dict) -> List[str]:
        """
        Detects signals a human interviewer would miss:
        - Response delay on technical questions
        - Repeated generic phrases
        - Confidence drop (early vs late scores)
        - Surface-level answer pattern
        """
        flags = []
        scores  = sess.get("scores", [])
        timings = sess.get("timings", [])

        # ── 1. Response delay on technical Qs (Q3=idx2, Q5=idx4) ────────────
        technical_turns = {3: 2, 5: 4}  # question_number → answers list index
        for qn, idx in technical_turns.items():
            if idx < len(timings) and timings[idx] > 8.0:
                flags.append(
                    f"Unusual response delay on Q{qn} ({timings[idx]:.1f}s)"
                )

        # ── 2. Repeated generic phrases (LLM call) ───────────────────────────
        transcript = " | ".join(sess.get("answers", []))
        if transcript:
            try:
                prompt = f"""Review these interview answers and list any phrases repeated 3+ times or generic filler reused across multiple answers.
Answers: {transcript[:2000]}
Return JSON only: {{"repeated_phrases": [...], "generic_filler_detected": true or false}}"""
                raw, _ = self.qwen.prompt(prompt)
                result = _extract_json(raw)
                if result.get("generic_filler_detected"):
                    repeated = result.get("repeated_phrases", [])
                    if repeated:
                        flags.append(f"Repeated filler phrases detected: {', '.join(repeated[:3])}")
            except Exception:
                pass

        # ── 3. Confidence drop: early avg vs late avg ────────────────────────
        if len(scores) >= 5:
            early_avg = mean(scores[:3])
            late_avg  = mean(scores[3:])
            if early_avg - late_avg > 2.5:
                flags.append(
                    f"Confidence drop detected: early avg {early_avg:.1f} vs late avg {late_avg:.1f}"
                )

        # ── 4. Surface-level pattern ─────────────────────────────────────────
        score_objs = sess.get("score_objects", [])
        surface_count = sum(1 for s in score_objs if s.get("depth") == "surface")
        if surface_count >= 4:
            flags.append(
                f"Predominantly surface-level answers ({surface_count}/{len(score_objs)} questions)"
            )

        # ── 5. Resume contradictions ─────────────────────────────────────────
        contradictions = [
            s["contradiction_detail"]
            for s in score_objs
            if s.get("resume_contradiction") and s.get("contradiction_detail")
        ]
        flags.extend(contradictions)

        return flags

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — POST-INTERVIEW REPORTS
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_phase3(self, session_id: str):
        """Entry point — called in a background thread after the interview ends."""
        sess = self.sessions.get(session_id)
        if not sess:
            return
        logger.info(f"[{session_id}] Phase 3: Generating reports...")
        try:
            report   = self._generate_hr_report(sess)
            feedback = self._generate_candidate_feedback(sess)
            sess["report"]   = report
            sess["feedback"] = feedback
            # ── Persist both files ───────────────────────────────────────────
            from services.transcript_service import TranscriptService
            import datetime
            ts = TranscriptService.__new__(TranscriptService)
            ts._ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts.save_hr_report(report)
            ts.save_candidate_report(feedback, sess["candidate"])
            logger.info(f"[{session_id}] ✓ Phase 3 complete.")
        except Exception as e:
            logger.error(f"[{session_id}] Phase 3 failed: {e}", exc_info=True)

    def _generate_hr_report(self, sess: dict) -> dict:
        """
        Aggregates scores per the 5 parameters, runs hidden signal capture,
        applies hard rules, calculates role fit %, and determines decision.
        """
        score_objs = sess.get("score_objects", [])
        scores_list = sess.get("scores", [])

        def _get(param: str):
            """Average of all scoring objects matching this parameter."""
            vals = [s["score"] for s in score_objs if s.get("parameter") == param]
            return round(mean(vals), 1) if vals else 5.0

        def _avg(*params):
            vals = [s["score"] for s in score_objs if s.get("parameter") in params]
            return round(mean(vals), 1) if vals else 5.0

        param_scores = {
            "resume_authenticity": _get("Resume Authenticity"),
            "conceptual_clarity":  _get("Conceptual Clarity"),
            "role_relevance":      _get("Role Relevance"),
            "problem_solving":     _get("Problem Solving"),
            "communication":       _avg("Communication"),
        }

        red_flags = self._capture_hidden_signals(sess)

        # ── Role fit % ───────────────────────────────────────────────────────
        role_fit_pct = round((sum(param_scores.values()) / 50.0) * 100.0, 1)

        if role_fit_pct >= 70:
            decision = "Shortlist"
        elif role_fit_pct >= 50:
            decision = "HR Review"
        else:
            decision = "Reject"

        # ── Strongest signal ─────────────────────────────────────────────────
        top = max(score_objs, key=lambda s: s.get("score", 0), default={})
        strongest_signal = top.get("key_signal", "") or "No strong signals detected."

        return {
            "candidate_name":  "Candidate",
            "role":            sess.get("role", ""),
            "role_fit_pct":    role_fit_pct,
            "decision":        decision,
            "scores":          param_scores,
            "red_flags":       red_flags or ["None detected"],
            "strongest_signal": strongest_signal,
        }

    def _generate_candidate_feedback(self, sess: dict) -> str:
        """
        Calls Qwen to generate 7-8 personalised improvement points.
        Warm, constructive tone — never reveals scores or pass/fail.
        """
        score_objs = sess.get("score_objects", [])
        red_flags  = self._capture_hidden_signals(sess)

        answer_quality_list = [
            f"Q{i+1}: {s.get('answer_quality', '?')} (depth: {s.get('depth','?')})"
            for i, s in enumerate(score_objs)
        ]
        key_signals = [s.get("key_signal", "") for s in score_objs if s.get("key_signal")]

        prompt = f"""You are a career coach writing post-interview feedback for a job candidate.

Role they applied for: {sess.get("role", "the role")}
Answer quality per question: {answer_quality_list}
Key signals observed: {key_signals}
Red flags detected: {red_flags}
Resume summary: {sess["candidate"].get("resume_summary", "")}

Write 7-8 improvement points. Rules:
- Specific to what they actually said — no generic advice
- Start each point with a positive framing before the suggestion
- Do not mention scores, numbers, or pass/fail
- Do not mention SwitchIt-Pro's internal evaluation process
- Cover: technical depth, communication, resume clarity, problem-solving approach
- 2-3 sentences per point max
- Tone: warm, honest, constructive

Return as a numbered list, plain text only."""

        try:
            raw, _ = self.qwen.prompt(prompt)
            # Strip any JSON leakage, return plain text
            return raw.strip()
        except Exception as e:
            logger.error(f"Candidate feedback generation failed: {e}")
            return (
                "1. Continue building depth in your technical answers — try to explain the 'why' behind your decisions.\n"
                "2. Reference specific projects from your resume when answering experience questions.\n"
                "3. Structure answers using Situation, Task, Action, Result (STAR) for clarity.\n"
                "4. Pause briefly before technical questions to collect your thoughts.\n"
                "5. Connect your experience explicitly to the role you are applying for.\n"
                "6. Reduce filler phrases to sound more confident and polished.\n"
                "7. Walk through your problem-solving steps explicitly — your reasoning matters.\n"
                "8. Close with a forward-looking statement connecting your goals to the role."
            )

    # ── Getters for server.py / WebSocket ────────────────────────────────────

    def get_report(self, session_id: str) -> Optional[dict]:
        sess = self.sessions.get(session_id)
        return sess.get("report") if sess else None

    def get_candidate_feedback(self, session_id: str) -> Optional[str]:
        sess = self.sessions.get(session_id)
        return sess.get("feedback") if sess else None
