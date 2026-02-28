"""
src/interview_runner.py
-----------------------
Orchestrates the complete interview session.

Full turn flow:
  1. Get next evaluation area to target (by weight priority)
  2. Query ChromaDB for best matching questions (OpenAI RAG retrieval)
  3. OpenAI selects the best question + optionally rephrases it
  4. Print the question (in a real system: speak to candidate)
  5. Receive candidate's response (CLI input or simulated)
  6. OpenAI scores the response (0-10 with rationale)
  7. OpenAI meta-evaluates: was the question good? was the score fair?
  8. Update state, print live diagnostics
  9. Repeat until all areas are covered

Components:
  OpenAI → all steps (interviewer + meta-evaluator + simulator)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .config import config
from .vector_store import VectorStore
from .rag_engine import RAGEngine
from .embedder import Embedder
from .openai_interviewer import QwenInterviewer
from .openai_evaluator import OpenAIMetaEvaluator
from .interview_state import InterviewStateManager
from .candidate_simulator import CandidateSimulator

logger = logging.getLogger(__name__)


class InterviewRunner:
    """
    Runs a complete Scout AI Interview session.

    OpenAI (Interviewer)      = question selection + scoring
    OpenAI (Meta-Evaluator)   = auditing interviewer question quality + scoring accuracy
    OpenAI (Simulator)        = simulating candidate responses (in --simulate mode)

    Usage
    -----
    runner = InterviewRunner()
    runner.setup_session(
        role="Sales Executive",
        experience_level="Mid-level",
        areas=[
            {"area": "Objection Handling",    "weight": 25, "questions": 2,
             "min_score": 6.0, "non_negotiable": True},
            {"area": "Communication Skills",   "weight": 20, "questions": 2},
            {"area": "Pipeline Management",    "weight": 20, "questions": 2},
            {"area": "Client Relationship",    "weight": 20, "questions": 2},
            {"area": "Resilience & Grit",      "weight": 15, "questions": 2},
        ],
        use_meta_eval=True,   # OpenAI meta-evaluates the interviewer (default: True)
    )
    runner.run_interactive()  # CLI interview
    """

    def __init__(self):
        self._store: Optional[VectorStore] = None
        self._engine: Optional[RAGEngine] = None
        self._qwen: Optional[QwenInterviewer] = None
        self._meta_evaluator: Optional[OpenAIMetaEvaluator] = None
        self._simulator: Optional[CandidateSimulator] = None
        self._state_mgr = InterviewStateManager(max_history=config.MAX_HISTORY)
        self._role = ""
        self._level = ""
        self._use_meta_eval = True
        self._simulate = False

    # ── Setup ─────────────────────────────────────────────────────

    def setup_session(
        self,
        role: str,
        experience_level: str,
        areas: List[Dict],
        use_meta_eval: bool = True,
        simulate: bool = False,
        persona: str = "average",
    ) -> None:
        """
        Initialize all components and start a session.

        Parameters
        ----------
        role             : job role being interviewed for
        experience_level : Fresher | Early Career | Mid-level | Senior
        areas            : list of {area, weight, questions, min_score?, non_negotiable?}
        use_meta_eval    : if True, run OpenAI meta-evaluation after each Qwen score
        simulate         : if True, OpenAI simulates candidate responses automatically
        persona          : candidate persona — 'strong' | 'average' | 'weak' | custom text
        """
        self._role = role
        self._level = experience_level
        self._use_meta_eval = use_meta_eval
        self._simulate = simulate

        print("\n" + "═" * 65)
        print("  SCOUT AI INTERVIEWER — Initializing")
        print("═" * 65)

        # ── Step 1: connect to ChromaDB ───────────────────────────
        print("\n  [1/3] Loading ChromaDB vector store...")
        embedder = Embedder()
        self._store = VectorStore(embedder=embedder)

        if not self._store.is_populated:
            raise RuntimeError(
                "ChromaDB is empty. Run load_data.py first:\n"
                "  python scripts/load_data.py"
            )

        print(f"       ✓ ChromaDB ready — {self._store.count} questions indexed")

        # ── Step 2: initialize OpenAI Interviewer ─────────────────
        print("\n  [2/3] Connecting to OpenAI Interviewer...")
        self._engine = RAGEngine(self._store)
        self._qwen = QwenInterviewer()
        print(f"       ✓ OpenAI interviewer ready ({config.OPENAI_INTERVIEWER_MODEL})")

        # ── Step 3: initialize OpenAI meta-evaluator + simulator ───
        if use_meta_eval or simulate:
            print("\n  [3/3] Connecting to OpenAI...")
            try:
                if use_meta_eval:
                    self._meta_evaluator = OpenAIMetaEvaluator()
                    print("       ✓ OpenAI meta-evaluator ready")
                if simulate:
                    self._simulator = CandidateSimulator(persona=persona)
                    print(f"       ✓ Candidate simulator ready  (persona: {self._simulator.persona_name})")
            except ValueError as e:
                print(f"       ⚠️  OpenAI not configured: {e}")
                self._use_meta_eval = False
                self._simulate = False
        else:
            print("\n  [3/3] Meta-evaluation disabled (--no-meta-eval)")

        # ── Start session state ────────────────────────────────────
        self._state_mgr.start_session(
            role=role,
            experience_level=experience_level,
            areas=areas,
        )

        total_questions = sum(a.get("questions", 2) for a in areas)
        print(f"\n  ✓ Session ready — {len(areas)} evaluation areas, {total_questions} questions")

    # ── Run Modes ─────────────────────────────────────────────────

    def run_interactive(self) -> None:
        """
        Interactive CLI interview — candidate types their answers.
        """
        state = self._state_mgr.session
        total_q = sum(t.questions_to_ask for t in state.area_tracker.values())
        consecutive_failures: dict = {}

        print("\n" + "═" * 65)
        print(f"  SCOUT AI INTERVIEW — {self._role}")
        print(f"  Level: {self._level}")
        print(f"  Total Questions: {total_q}")
        print(f"  Interviewer: {config.OPENAI_INTERVIEWER_MODEL} | Meta-Eval: {'ON' if self._use_meta_eval else 'OFF'}")
        print("═" * 65)
        print()
        print("  👋 Hi there! Thanks so much for joining us today.")
        print(f"  We're excited to learn more about you for the {self._role} role.")
        print("  This will be a relaxed conversation — just be yourself, take your")
        print("  time with each answer, and feel free to share real examples.")
        print(f"  We'll go through {total_q} questions across a few key areas.")
        print("  Let's get started!")
        print()

        # ── Simulation mode ───────────────────────────────────────
        if self._simulate:
            self.run_simulated()
            return

        while not self._state_mgr.is_complete():
            area = self._state_mgr.get_next_area()
            if not area:
                break

            tracker = state.area_tracker[area]
            turn_num = state.total_turns + 1
            print(f"\n{'─' * 65}")
            print(f"  Question {turn_num} of {total_q}  ·  Topic: {area}")
            print(f"{'─' * 65}")

            try:
                result = self._run_turn(area=area, turn_num=turn_num, mode="interactive")
                consecutive_failures[area] = 0
            except Exception as e:
                consecutive_failures[area] = consecutive_failures.get(area, 0) + 1
                logger.error("Turn failed: %s", e)
                print(f"\n  ❌ Error: {e}")
                if consecutive_failures[area] >= 3:
                    print(f"  ⚠️  Could not get a question for '{area}' after 3 attempts. Moving on.")
                    # Mark area as complete so the loop advances
                    tracker.questions_asked = tracker.questions_to_ask
                else:
                    print("  Retrying...")
                continue

        # Final scorecard
        print(self._state_mgr.get_live_scorecard())

    def run_simulated(self) -> None:
        """
        Fully automated interview — OpenAI generates candidate responses.
        No human input required. Use for pipeline testing.
        """
        state = self._state_mgr.session
        total_q = sum(t.questions_to_ask for t in state.area_tracker.values())
        persona = self._simulator.persona_name if self._simulator else "unknown"
        consecutive_failures: dict = {}

        print("\n" + "═" * 65)
        print(f"  SCOUT AI INTERVIEW — SIMULATE MODE")
        print(f"  Role: {self._role} | Level: {self._level}")
        print(f"  Candidate Persona: {persona.upper()}")
        print("  Responses: Generated by OpenAI")
        print("═" * 65 + "\n")

        while not self._state_mgr.is_complete():
            area = self._state_mgr.get_next_area()
            if not area:
                break

            tracker = state.area_tracker[area]
            turn_num = state.total_turns + 1
            print(f"\n{'─' * 65}")
            print(f"  Question {turn_num}/{total_q}  |  Area: {area}")
            print(f"{'─' * 65}")

            try:
                self._run_turn(area=area, turn_num=turn_num, mode="simulate")
                consecutive_failures[area] = 0
            except Exception as e:
                consecutive_failures[area] = consecutive_failures.get(area, 0) + 1
                logger.error("Turn failed: %s", e)
                print(f"\n  ❌ Error: {e}")
                if consecutive_failures[area] >= 3:
                    print(f"  ⚠️  Could not get a question for '{area}' after 3 attempts. Moving on.")
                    tracker.questions_asked = tracker.questions_to_ask
                continue

        print(self._state_mgr.get_live_scorecard())

    def run_demo(self, candidate_responses: List[str]) -> None:
        """
        Demo/test mode — inject pre-written candidate responses.
        """
        state = self._state_mgr.session
        total_q = sum(t.questions_to_ask for t in state.area_tracker.values())
        response_idx = 0

        print("\n" + "═" * 65)
        print(f"  SCOUT AI INTERVIEW — DEMO MODE")
        print(f"  Role: {self._role} | Level: {self._level}")
        print("═" * 65 + "\n")

        while not self._state_mgr.is_complete():
            area = self._state_mgr.get_next_area()
            if not area:
                break

            candidate_answer = (
                candidate_responses[response_idx]
                if response_idx < len(candidate_responses)
                else "I don't have a specific example for that."
            )
            response_idx += 1

            try:
                self._run_turn(
                    area=area,
                    turn_num=state.total_turns + 1,
                    mode="demo",
                    demo_response=candidate_answer,
                )
            except Exception as e:
                logger.error("Turn failed: %s", e)
                print(f"\n  ❌ Error in turn: {e}")

        print(self._state_mgr.get_live_scorecard())

    # ── Core Turn Logic ───────────────────────────────────────────

    def _run_turn(
        self,
        area: str,
        turn_num: int,
        mode: str = "interactive",
        demo_response: str = "",
    ) -> Dict:
        """
        Execute one complete interview turn:
          1. RAG: retrieve question candidates (Qwen embeddings + ChromaDB)
          2. Qwen: select best question from candidates
          3. Get candidate's response
          4. Qwen: score the response
          5. OpenAI: meta-evaluate Qwen's question + scoring
          6. Update state

        Returns dict with full turn details.
        """
        session = self._state_mgr.session

        # ── 1. RAG Retrieval ──────────────────────────────────────────────
        # Fallback chain: narrow → broader → no-filter → large pool
        candidates = self._engine.search_for_parameter(
            evaluation_area=area,
            role=self._role,
            experience_level=self._level,
            top_k=config.TOP_K,
        )

        if not candidates:
            candidates = self._engine.search_for_parameter(
                evaluation_area=area,
                role=self._role,
                top_k=config.TOP_K,
            )

        if not candidates:
            candidates = self._engine.search_for_parameter(
                evaluation_area=area,
                role=self._role,
                top_k=config.TOP_K * 4,
            )

        if not candidates:
            raise ValueError(
                f"No questions found for '{area}' in ChromaDB. "
                f"Check that questions_store.xlsx has questions for role='{self._role}'."
            )

        # If all top-K candidates are already asked, fetch a much larger pool
        unasked = [c for c in candidates if c.get("id") not in session.asked_question_ids]
        if not unasked:
            logger.info(
                "All top-%d candidates for '%s' already asked. Expanding search pool...",
                len(candidates), area,
            )
            candidates = self._engine.search_for_parameter(
                evaluation_area=area,
                role=self._role,
                top_k=config.TOP_K * 6,   # expand to 48 candidates
            )
            unasked = [c for c in candidates if c.get("id") not in session.asked_question_ids]

        if not unasked:
            raise ValueError(
                f"No available questions for '{area}'. "
                f"All {len(candidates)} retrieved questions have already been asked."
            )

        # ── 2. Qwen: Select the best question ─────────────────────
        context = self._state_mgr.get_context_string()
        selected = self._qwen.select_question(
            candidates=candidates,
            role=self._role,
            experience_level=self._level,
            evaluation_area=area,
            conversation_context=context,
            asked_ids=session.asked_question_ids,
        )

        question_text = selected["question_text"]
        question_id   = selected["question_id"]
        rag_context   = selected["rag_context"]

        print(f"\n  🤖 AI: {question_text}")
        if selected.get("rephrased"):
            print(f"     ↳ (rephrased: {selected.get('reason', '')})")

        # ── 3. Candidate's response ───────────────────────────────
        if mode == "interactive":
            print()
            candidate_response = input("  👤 You: ").strip()
            if not candidate_response:
                candidate_response = "(No response provided)"
        elif mode == "simulate" and self._simulator:
            print(f"\n  ⏳ Simulating candidate response ({self._simulator.persona_name} persona)...")
            candidate_response = self._simulator.answer(
                question=question_text,
                role=self._role,
                experience_level=self._level,
                evaluation_area=area,
                conversation_context=context,
            )
            print(f"\n  🎭 Simulated [{self._simulator.persona_name.upper()}]: {candidate_response}")
        else:
            candidate_response = demo_response
            print(f"\n  👤 Candidate: {candidate_response}")

        # ── 4. Qwen: Score the response ───────────────────────────
        print("\n  ⏳ AI is scoring the response...")
        scoring = self._qwen.score_response(
            question_text=question_text,
            candidate_response=candidate_response,
            evaluation_area=area,
            what_ai_listens_for=rag_context.get("what_ai_listens_for", ""),
            strong_signal_example=rag_context.get("strong_signal_example", ""),
            weak_signal_example=rag_context.get("weak_signal_example", ""),
            role=self._role,
            experience_level=self._level,
            conversation_context=context,
        )

        qwen_score = scoring["raw_score"]
        signal     = scoring["signal_strength"].upper()
        score_icon = "✅" if qwen_score >= 7 else ("⚠️" if qwen_score >= 5 else "❌")

        print(f"\n  📊 AI Score: {qwen_score}/10  {score_icon}  [{signal}]")
        print(f"     Rationale: {scoring['rationale']}")

        if scoring.get("strengths"):
            print(f"     Strengths: {', '.join(scoring['strengths'][:2])}")
        if scoring.get("weaknesses"):
            print(f"     Weaknesses: {', '.join(scoring['weaknesses'][:2])}")

        # ── 5. Record turn in state ───────────────────────────────
        turn = self._state_mgr.record_turn(
            question_id=question_id,
            question_text=question_text,
            candidate_response=candidate_response,
            evaluation_area=area,
            qwen_score=qwen_score,
            qwen_rationale=scoring["rationale"],
            qwen_strengths=scoring.get("strengths", []),
            qwen_weaknesses=scoring.get("weaknesses", []),
            qwen_rephrased=selected.get("rephrased", False),
            qwen_select_reason=selected.get("reason", ""),
        )

        # ── 6. OpenAI Meta-Evaluation ─────────────────────────────
        if self._use_meta_eval and self._meta_evaluator:
            print("\n  ⏳ OpenAI is meta-evaluating the interviewer's performance...")
            try:
                meta = self._meta_evaluator.evaluate_turn(
                    turn_number=turn.turn_number,
                    question_id=question_id,
                    question_text=question_text,
                    evaluation_area=area,
                    role=self._role,
                    experience_level=self._level,
                    candidate_response=candidate_response,
                    qwen_score=qwen_score,
                    qwen_rationale=scoring["rationale"],
                    qwen_strengths=scoring.get("strengths", []),
                    qwen_weaknesses=scoring.get("weaknesses", []),
                    rag_context=rag_context,
                )

                self._state_mgr.update_meta_eval(
                    turn_number=turn.turn_number,
                    question_verdict=meta.question_quality.verdict,
                    question_quality=meta.question_quality.overall_quality,
                    openai_score=meta.scoring_accuracy.openai_score,
                    scoring_verdict=meta.scoring_accuracy.verdict,
                    score_delta=meta.scoring_accuracy.score_delta,
                    openai_assessment=meta.scoring_accuracy.assessment,
                    alerts=meta.alerts,
                )

                meta.print_diagnostics()

            except Exception as e:
                logger.error("OpenAI meta-evaluation failed: %s", e)
                print(f"\n  ⚠️  Meta-evaluation error: {e}")

        # ── 7. Show live scorecard progress ───────────────────────
        tracker = session.area_tracker.get(area)
        if tracker:
            progress = f"{tracker.questions_asked}/{tracker.questions_to_ask}"
            print(f"\n  📈 {area}: {tracker.display} | Progress: {progress}")

        return {
            "turn_number": turn.turn_number,
            "question_id": question_id,
            "question_text": question_text,
            "candidate_response": candidate_response,
            "evaluation_area": area,
            "qwen_score": qwen_score,
            "qwen_signal": scoring["signal_strength"],
        }

    # ── State Access ──────────────────────────────────────────────

    def get_session(self):
        return self._state_mgr.session

    def save_session(self, filepath: str) -> None:
        self._state_mgr.save_to_file(filepath)
