"""
services/rag_service.py
-----------------------
RAG (Retrieval-Augmented Generation) Service for conversational-ai-scouts.

NEW DESIGN — Structured Interview Mode:
  fetch_interview_questions(role, n) is called ONCE at interview start.
  It performs semantic search on the scout_ai_interviewer ChromaDB,
  returning n curated questions ranked by relevance to the given role.

  The question list is structured as:
    [0]     intro      — "Tell me about yourself"
    [1..n-2] domain   — semantically retrieved from VectorDB
    [n-1]  outro      — closing / wrap-up

  The LLM service receives this queue and asks one question per turn,
  advancing through the list naturally.

Architecture:
  - Connects to scout_ai_interviewer/chroma_store (read-only)
  - Embeds queries using nomic-embed-text via Ollama
  - Returns top-K semantically relevant questions on startup

Usage:
  rag = RAGService()
  questions = rag.fetch_interview_questions(role="Software Engineer", n=5)
  # questions → ["Tell me about yourself.", "...", "...", "...", "Wrap up / any questions?"]
"""

import logging
import random
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Absolute path to the scout_ai_interviewer chroma_store ────────────────────
_SCOUT_CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "scout_ai_interviewer" / "chroma_store"
_COLLECTION_NAME  = "scout_questions"

# ── Ollama embedding endpoint (same model used when indexing: nomic-embed-text) ─
_OLLAMA_URL      = "http://localhost:11434"
_EMBEDDING_MODEL = "nomic-embed-text"

# ── Default intro / outro templates (used when no good DB match found) ─────────
_INTRO_QUESTIONS = [
    "Great to meet you! Could you start by telling me a little about yourself and your background?",
    "Welcome! To kick things off, could you walk me through your professional journey so far?",
    "Hi, thanks for joining us today! Please go ahead and introduce yourself.",
]

_OUTRO_QUESTIONS = [
    "We're coming to the end of our session. Do you have any questions for me or for the team at SwitchIt?",
    "That wraps up my questions. Is there anything you'd like to ask about the role or the company?",
    "Thanks so much for your time today! Before we close, do you have any questions for us?",
]


class RAGService:
    """
    Retrieval-Augmented Generation bridge to the scout_ai_interviewer vector DB.

    Structured interview flow:
      1. At startup: embed the role description and search ChromaDB for the
         most semantically relevant interview questions for that role.
      2. Return a fixed ordered list: intro → domain questions → outro.
      3. The LLM service walks through this list turn-by-turn.

    This gives the conversational AI a focused, curated question plan instead
    of ad-hoc retrieval on every utterance.
    """

    def __init__(self):
        self._client     = None
        self._collection = None
        self._ready      = False
        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        """
        Open a read-only connection to the scout_ai_interviewer ChromaDB.
        Fails gracefully — if ChromaDB is unavailable the AI still works
        (it will use hardcoded fallback questions).
        """
        try:
            import chromadb
            from chromadb.config import Settings

            if not _SCOUT_CHROMA_DIR.exists():
                logger.warning(
                    "scout_ai_interviewer chroma_store not found at: %s\n"
                    "RAG will be disabled. Make sure scout_ai_interviewer is set up "
                    "and the chroma_store has been populated.",
                    _SCOUT_CHROMA_DIR,
                )
                return

            self._client = chromadb.PersistentClient(
                path=str(_SCOUT_CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_collection(name=_COLLECTION_NAME)
            doc_count = self._collection.count()

            if doc_count == 0:
                logger.warning(
                    "ChromaDB collection '%s' is empty. "
                    "Run the scout_ai_interviewer ingestion script first.",
                    _COLLECTION_NAME,
                )
                return

            self._ready = True
            logger.info(
                "✓ RAG Service connected to scout_ai_interviewer ChromaDB "
                "('%s', %d questions)",
                _COLLECTION_NAME, doc_count,
            )

        except ImportError:
            logger.warning("chromadb not installed. RAG disabled. Run: pip install chromadb")
        except Exception as e:
            logger.warning("RAG Service failed to connect to ChromaDB: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_interview_questions(
        self,
        role: str = "Software Engineer",
        n: int = 6,
        experience_level: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> List[str]:
        """
        Fetch a structured list of n interview questions for the given role.

        Performs a ONE-TIME semantic search against the ChromaDB at interview
        start. Returns an ordered list:
            [0]       Intro question  (greeting + "tell me about yourself")
            [1..n-2]  Domain questions semantically retrieved from VectorDB
            [n-1]     Outro question  (wrap-up + "any questions for us?")

        If VectorDB is unavailable, returns a sensible hardcoded fallback list.

        Parameters
        ----------
        role             : candidate role, e.g. "Software Engineer"
        n                : total number of questions (min 3, max 10)
        experience_level : optional filter, e.g. "Mid-level"
        industry         : optional filter, e.g. "Technology", "Finance"

        Returns
        -------
        List[str] — ordered question texts ready to ask
        """
        n = max(3, min(n, 10))                 # clamp to [3, 10]
        num_domain = n - 2                     # slots between intro and outro

        intro = random.choice(_INTRO_QUESTIONS)
        outro = random.choice(_OUTRO_QUESTIONS)

        domain_questions = self._fetch_domain_questions(
            role=role,
            experience_level=experience_level,
            industry=industry,
            n=num_domain,
        )

        question_list = [intro] + domain_questions + [outro]

        logger.info(
            "✓ Interview question plan loaded (%d total) "
            "for role='%s', industry='%s', level='%s':\n%s",
            len(question_list),
            role,
            industry or "any",
            experience_level or "any",
            "\n".join(f"  Q{i+1}: {q[:80]}..." if len(q) > 80 else f"  Q{i+1}: {q}"
                      for i, q in enumerate(question_list)),
        )

        return question_list

    def is_ready(self) -> bool:
        """Returns True if the vector DB is connected and ready."""
        return self._ready

    # ── Domain Question Retrieval ─────────────────────────────────────────────

    def _fetch_domain_questions(
        self,
        role: str,
        experience_level: Optional[str],
        n: int,
        industry: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve n domain-specific questions from ChromaDB via semantic search.

        Filters by industry, role, and experience_level so results are tightly
        scoped to the candidate's profile. Falls back to hardcoded questions if
        VectorDB is unavailable or returns too few results.
        """
        if not self._ready:
            logger.warning("VectorDB unavailable — using fallback domain questions.")
            return self._fallback_domain_questions(role, n)

        try:
            # Build a rich semantic query that incorporates all three dimensions
            parts = ["technical interview question"]
            if industry:
                parts.append(f"in the {industry} industry")
            if role:
                parts.append(f"for {role}")
            if experience_level:
                parts.append(f"at {experience_level} level")
            query = " ".join(parts)

            embedding = self._embed(query)
            if not embedding:
                return self._fallback_domain_questions(role, n)

            # Build a ChromaDB $and metadata filter from whichever fields are set
            where = self._build_filter(
                role=role,
                experience_level=experience_level,
                industry=industry,
            )

            kwargs: Dict[str, Any] = dict(
                query_embeddings=[embedding],
                n_results=min(n * 3, self._collection.count()),  # fetch extra for dedup
                include=["documents", "distances", "metadatas"],
            )
            if where:
                kwargs["where"] = where

            raw = self._collection.query(**kwargs)
            results = self._parse_results(raw)

            if not results:
                logger.warning("ChromaDB returned no results for role='%s'.", role)
                return self._fallback_domain_questions(role, n)

            # Extract clean question texts, deduplicate, take top n
            questions = []
            seen: set = set()
            for r in results:
                text = self._extract_question_text(r.get("document", ""))
                text_lower = text.lower()
                if text and text_lower not in seen:
                    seen.add(text_lower)
                    questions.append(text)
                if len(questions) == n:
                    break

            # Pad with fallback if we got fewer than needed
            if len(questions) < n:
                fallbacks = self._fallback_domain_questions(role, n - len(questions))
                questions.extend(fallbacks)

            return questions[:n]

        except Exception as e:
            logger.warning("RAG domain question retrieval failed: %s", e)
            return self._fallback_domain_questions(role, n)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Embed text using nomic-embed-text via Ollama.
        The SAME model used to index the scout_ai_interviewer data,
        ensuring the vector space is compatible for semantic search.
        """
        try:
            response = requests.post(
                f"{_OLLAMA_URL}/api/embeddings",
                json={"model": _EMBEDDING_MODEL, "prompt": text},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" not in data:
                raise RuntimeError(f"Ollama response missing 'embedding' key: {data}")
            return data["embedding"]
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama not running at %s — RAG embedding skipped. "
                "Start Ollama with: ollama serve",
                _OLLAMA_URL,
            )
            return None
        except Exception as e:
            logger.warning("Embedding error: %s", e)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(raw: dict) -> List[Dict]:
        """Flatten raw ChromaDB query response into clean result dicts."""
        results = []
        ids       = raw.get("ids",       [[]])[0]
        docs      = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metas     = raw.get("metadatas", [[]])[0]

        for qid, doc, dist, meta in zip(ids, docs, distances, metas):
            results.append({
                "id":         qid,
                "document":   doc,
                "similarity": round(1 - dist, 4),
                "metadata":   meta or {},
            })
        return results

    @staticmethod
    def _extract_question_text(doc_text: str) -> str:
        """
        Extract the clean question text from a ChromaDB document.

        Documents are stored as:
          "Question: ... | Role: ... | Area: ..."
        We only want the question part.
        """
        doc_text = doc_text.strip()
        if "Question:" in doc_text:
            q_start = doc_text.index("Question:") + len("Question:")
            q_end   = doc_text.index("|") if "|" in doc_text else len(doc_text)
            return doc_text[q_start:q_end].strip()
        return doc_text

    @staticmethod
    def _build_filter(
        role: Optional[str] = None,
        experience_level: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Build a ChromaDB metadata filter dict from the candidate's profile.

        Supports any combination of industry, role, and experience_level.
        - 0 fields → None (no filter, search everything)
        - 1 field  → simple {key: value}
        - 2+ fields → {"$and": [{key: value}, ...]}

        ChromaDB metadata keys used:
            industry         → "industry"
            role             → "role"
            experience_level → "experience_level"
        """
        conditions = []
        if industry:
            conditions.append({"industry": industry})
        if role:
            conditions.append({"role": role})
        if experience_level:
            conditions.append({"experience_level": experience_level})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _fallback_domain_questions(role: str, n: int) -> List[str]:
        """
        Hardcoded fallback questions used when ChromaDB is unavailable.
        Generic enough to cover any role, specific enough to be useful.
        """
        pool = [
            f"Can you walk me through a recent project you're particularly proud of in your {role} career?",
            "What's a challenging technical problem you've solved, and how did you approach it?",
            f"How do you stay up to date with the latest trends and best practices in {role}?",
            "Describe a situation where you had to work under tight deadlines. How did you manage it?",
            "Tell me about a time you disagreed with a team member. How did you handle it?",
            "What does your typical decision-making process look like when you face a complex problem?",
            "Can you describe your experience with cross-functional team collaboration?",
            "What tools, frameworks, or methodologies do you rely on most in your day-to-day work?",
        ]
        # Shuffle for variety, then take n
        random.shuffle(pool)
        return pool[:n]
