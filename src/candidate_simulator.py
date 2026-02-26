"""
src/candidate_simulator.py
--------------------------
Simulates a job candidate's responses using OpenAI GPT.

Used when running the interview in --simulate mode (no human needed).
OpenAI generates realistic candidate answers based on a persona.

Personas:
  strong    : Highly experienced, gives detailed STAR answers, concrete metrics
  average   : Moderate experience, gives okay answers, some gaps
  weak      : Entry-level, vague answers, no structured approach
  custom    : You provide a custom description

Usage
-----
sim = CandidateSimulator(persona="strong")
response = sim.answer(
    question="How do you handle price objections?",
    role="Sales Executive",
    experience_level="Mid-level",
    evaluation_area="Objection Handling",
)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .config import config

logger = logging.getLogger(__name__)

PERSONAS = {
    "strong": (
        "You are a highly experienced Sales Executive with 8 years in B2B SaaS sales. "
        "You have a proven record of exceeding quotas by 130%+. "
        "You give structured, specific STAR-format answers with real metrics and examples. "
        "You are articulate, confident, and back up every claim with data."
    ),
    "average": (
        "You are a Sales Executive with 3 years of experience. "
        "You have performed adequately but haven't been a top performer. "
        "Your answers are reasonable but sometimes vague. "
        "You occasionally struggle to give specific metrics or structured examples."
    ),
    "weak": (
        "You are a junior Sales Executive with less than 1 year of experience. "
        "You are eager but inexperienced. "
        "Your answers are generic, rambling, and lack structure or specific examples. "
        "You often say things like 'I would try to...' instead of 'I have done...'"
    ),
}


class CandidateSimulator:
    """
    Simulates a job candidate using OpenAI.

    This is the ONLY other place OpenAI is used (besides the meta-evaluator).
    It is used in --simulate mode to auto-generate candidate responses for
    pipeline testing — no human typing needed.

    Parameters
    ----------
    persona : "strong" | "average" | "weak" | custom description string
    """

    def __init__(self, persona: str = "average"):
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set in .env.\n"
                "Add: OPENAI_API_KEY=sk-proj-..."
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_EVAL_MODEL   # reuse same model

        # Build persona description
        if persona in PERSONAS:
            self._persona_desc = PERSONAS[persona]
            self._persona_name = persona
        else:
            # Custom persona — user passed their own description
            self._persona_desc = persona
            self._persona_name = "custom"

        logger.info(
            "CandidateSimulator ready — persona: %s, model: %s",
            self._persona_name, self._model,
        )

    def answer(
        self,
        question: str,
        role: str,
        experience_level: str,
        evaluation_area: str,
        conversation_context: str = "",
    ) -> str:
        """
        Generate a candidate response to the given interview question.

        Parameters
        ----------
        question            : question asked by Qwen interviewer
        role                : job role (e.g. "Sales Executive")
        experience_level    : e.g. "Mid-level"
        evaluation_area     : what is being evaluated
        conversation_context: recent conversation for continuity

        Returns
        -------
        str: simulated candidate response (1-3 paragraphs)
        """
        system = (
            f"{self._persona_desc}\n\n"
            f"You are being interviewed for a {experience_level} {role} position. "
            f"Answer naturally, as a real candidate would speak. "
            f"Do NOT be perfect — match your persona's level. "
            f"Keep your response to 3-6 sentences unless the question requires more detail."
        )

        context_block = ""
        if conversation_context and "first question" not in conversation_context.lower():
            context_block = f"\n\nConversation so far:\n{conversation_context}\n"

        user_prompt = (
            f"{context_block}"
            f"Interviewer's question about {evaluation_area}:\n"
            f"\"{question}\"\n\n"
            f"Your answer (speak naturally as the candidate):"
        )

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system",  "content": system},
                        {"role": "user",    "content": user_prompt},
                    ],
                    temperature=0.85,   # higher = more varied, human-like answers
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                if attempt == 2:
                    logger.error("CandidateSimulator failed: %s", e)
                    return "I'd rather pass on this one."
                time.sleep(2.0 * (attempt + 1))

        return "I don't have a specific example for that."

    @property
    def persona_name(self) -> str:
        return self._persona_name

    @property
    def persona_desc(self) -> str:
        return self._persona_desc[:100] + "..."
