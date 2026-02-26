"""
src/rag_engine.py
-----------------
RAG Query Engine — high-level query interface over VectorStore.

Provides business-language methods built on top of VectorStore:
  - search_for_parameter()  — semantic search filtered by role + area
  - search_by_role()        — role-scoped semantic search
  - get_follow_ups()        — follow-up questions for a parent
  - random_sample()         — random pool for variety

All searches embed query text using Qwen-7B (same model that indexed the data).
"""

from __future__ import annotations

import random
import logging
from typing import Dict, List, Optional, Any

from .config import config
from .vector_store import VectorStore, QueryResult

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    High-level RAG query interface.

    Usage
    -----
    engine = RAGEngine(store)

    # Find questions for an evaluation area:
    results = engine.search_for_parameter(
        evaluation_area="Objection Handling",
        role="Sales Executive",
        experience_level="Mid-level",
        top_k=8,
    )

    # Role-scoped semantic search:
    results = engine.search_by_role(
        role="Sales Executive",
        query="closing a deal with a hesitant client",
        top_k=5,
    )
    """

    def __init__(self, store: Optional[VectorStore] = None):
        self._store = store or VectorStore()

    # ── Semantic Search Methods ────────────────────────────────────

    def search_for_parameter(
        self,
        evaluation_area: str,
        role: str,
        experience_level: Optional[str] = None,
        question_type: Optional[str] = None,
        top_k: int | None = None,
    ) -> List[QueryResult]:
        """
        Find the best questions for a specific evaluation parameter.

        Builds a semantic query from the evaluation area and filters
        by role and optionally experience_level.

        Parameters
        ----------
        evaluation_area  : e.g. "Objection Handling"
        role             : e.g. "Sales Executive"
        experience_level : optional, e.g. "Mid-level"
        question_type    : optional, e.g. "Behavioral"
        top_k            : number of results

        Returns
        -------
        List of ChromaDB result dicts.
        """
        # Build a richer semantic query from the evaluation area
        query = f"{evaluation_area} skills for {role}"
        if experience_level:
            query += f" {experience_level} level"

        filter_dict = self._build_filter(
            role=role,
            experience_level=experience_level,
            question_type=question_type,
        )

        results = self._store.search(
            query_text=query,
            top_k=top_k or config.TOP_K,
            where=filter_dict,
        )

        logger.debug(
            "RAG search for '%s' (role=%s, level=%s) → %d results",
            evaluation_area, role, experience_level, len(results),
        )
        return results

    def search_by_role(
        self,
        role: str,
        query: str,
        top_k: int | None = None,
        experience_level: Optional[str] = None,
        question_type: Optional[str] = None,
    ) -> List[QueryResult]:
        """Semantic search scoped to a specific role."""
        filter_dict = self._build_filter(
            role=role,
            experience_level=experience_level,
            question_type=question_type,
        )
        return self._store.search(
            query_text=query,
            top_k=top_k or config.TOP_K,
            where=filter_dict,
        )

    def get_follow_ups(self, parent_id: str, limit: int = 5) -> List[QueryResult]:
        """Retrieve follow-up questions for a given parent question ID."""
        return self._store.filter_by_metadata(
            where={"parent_question_id": parent_id},
            limit=limit,
        )

    def get_by_metadata(
        self,
        role: Optional[str] = None,
        evaluation_area: Optional[str] = None,
        experience_level: Optional[str] = None,
        question_type: Optional[str] = None,
        limit: int = 30,
    ) -> List[QueryResult]:
        """Pure metadata filter — no semantic scoring."""
        conditions = []
        if role:
            conditions.append({"role": role})
        if evaluation_area:
            conditions.append({"evaluation_area": evaluation_area})
        if experience_level:
            conditions.append({"experience_level": experience_level})
        if question_type:
            conditions.append({"question_type": question_type})

        if not conditions:
            raise ValueError("At least one filter must be specified.")

        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        return self._store.filter_by_metadata(where=where, limit=limit)

    def random_sample(
        self,
        role: str,
        experience_level: Optional[str] = None,
        n: int = 5,
        pool_size: int = 100,
    ) -> List[QueryResult]:
        """Return n randomly selected questions from a filtered pool."""
        conditions: List[Dict[str, Any]] = [{"role": role}]
        if experience_level:
            conditions.append({"experience_level": experience_level})

        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        pool = self._store.filter_by_metadata(where=where, limit=pool_size)

        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def get_by_id(self, question_id: str):
        return self._store.get_by_id(question_id)

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _build_filter(
        role: Optional[str],
        experience_level: Optional[str],
        question_type: Optional[str],
    ) -> Optional[Dict]:
        """Build a ChromaDB $and filter dict from optional fields."""
        conditions = []
        if role:
            conditions.append({"role": role})
        if experience_level:
            conditions.append({"experience_level": experience_level})
        if question_type:
            conditions.append({"question_type": question_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @property
    def store(self) -> VectorStore:
        return self._store
