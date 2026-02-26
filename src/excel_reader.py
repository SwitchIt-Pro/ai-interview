"""
src/excel_reader.py
--------------------
Reads questions_store.xlsx and yields structured Question objects.

Design:
  - openpyxl in read-only mode for memory efficiency
  - Returns Question dataclass objects (attribute access, not magic indices)
  - Skips rows where question_id or question_text is empty
  - All strings are stripped; numeric fields cast with fallback defaults
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List, Optional

import openpyxl

from .config import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────────────────────────

@dataclass
class Question:
    """One row from the Questions Store sheet."""

    question_id: str
    question_text: str
    industry: str = ""
    role: str = ""
    evaluation_area: str = ""
    experience_level: str = ""
    priority_level: str = ""
    question_type: str = ""
    question_depth: str = ""
    skill_clusters_tested: str = ""
    what_ai_listens_for: str = ""
    strong_signal_example: str = ""
    weak_signal_example: str = ""
    follow_up_trigger: str = ""
    parent_question_id: str = ""
    status: str = "Active"
    ctc_band: str = ""
    expected_duration_seconds: int = 0
    difficulty_score: int = 0
    usage_count: int = 0
    success_rate: float = 0.0
    variant_group_id: str = ""
    tags: str = ""

    def to_document(self) -> str:
        """
        Build the text corpus that will be embedded by Qwen and stored in ChromaDB.
        We include high-signal fields so semantic search can match on skills,
        evaluation area, role context, etc.
        """
        parts = [
            f"Question: {self.question_text}",
            f"Role: {self.role}",
            f"Evaluation Area: {self.evaluation_area}",
            f"Skills Tested: {self.skill_clusters_tested}",
            f"Question Type: {self.question_type}",
            f"What to listen for: {self.what_ai_listens_for}",
        ]
        return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())

    def to_metadata(self) -> dict:
        """
        Return a flat dict for ChromaDB metadata storage.
        ChromaDB only accepts str | int | float | bool values.
        Heavy text blobs (what_ai_listens_for, signal examples) are stored
        in metadata so the evaluator can retrieve them via RAG context.
        """
        d = asdict(self)
        safe = {}
        for k, v in d.items():
            if v is None:
                safe[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                safe[k] = v
            else:
                safe[k] = str(v)
        return safe


# ──────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────

class ExcelQuestionReader:
    """
    Reads questions_store.xlsx as an iterable of Question objects.

    Usage
    -----
    reader = ExcelQuestionReader()
    questions = reader.load_all()         # list[Question]
    for q in reader.iter_questions():     # streaming
        print(q.question_id, q.question_text)
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or config.EXCEL_FILE_PATH
        if not self.path.exists():
            raise FileNotFoundError(
                f"Excel file not found: {self.path}\n"
                f"Copy questions_store.xlsx into the 'data/' folder."
            )
        logger.info("ExcelQuestionReader ready: %s", self.path)

    @staticmethod
    def _s(val) -> str:
        return str(val).strip() if val is not None else ""

    @staticmethod
    def _i(val, default: int = 0) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _f(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _row_to_question(self, row: tuple) -> Optional[Question]:
        """Convert one openpyxl row tuple into a Question dataclass."""
        c = config.COLUMNS
        try:
            qid  = self._s(row[c["question_id"]])
            text = self._s(row[c["question_text"]])
            if not qid or not text:
                return None

            return Question(
                question_id=qid,
                question_text=text,
                industry=self._s(row[c["industry"]]),
                role=self._s(row[c["role"]]),
                evaluation_area=self._s(row[c["evaluation_area"]]),
                experience_level=self._s(row[c["experience_level"]]),
                priority_level=self._s(row[c["priority_level"]]),
                question_type=self._s(row[c["question_type"]]),
                question_depth=self._s(row[c["question_depth"]]),
                skill_clusters_tested=self._s(row[c["skill_clusters_tested"]]),
                what_ai_listens_for=self._s(row[c["what_ai_listens_for"]]),
                strong_signal_example=self._s(row[c["strong_signal_example"]]),
                weak_signal_example=self._s(row[c["weak_signal_example"]]),
                follow_up_trigger=self._s(row[c["follow_up_trigger"]]),
                parent_question_id=self._s(row[c["parent_question_id"]]),
                status=self._s(row[c["status"]]) or "Active",
                ctc_band=self._s(row[c["ctc_band"]]),
                expected_duration_seconds=self._i(row[c["expected_duration_seconds"]]),
                difficulty_score=self._i(row[c["difficulty_score"]]),
                usage_count=self._i(row[c["usage_count"]]),
                success_rate=self._f(row[c["success_rate"]]),
                variant_group_id=self._s(row[c["variant_group_id"]]),
                tags=self._s(row[c["tags"]]),
            )
        except IndexError as e:
            logger.warning("Row too short, skipping: %s", e)
            return None

    def iter_questions(self, active_only: bool = True) -> Iterator[Question]:
        """Stream questions from Excel one by one (memory efficient)."""
        wb = openpyxl.load_workbook(self.path, read_only=True, data_only=True)
        ws = wb[config.SHEET_NAME]
        yielded = skipped = 0

        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:  # skip header
                continue
            q = self._row_to_question(row)
            if q is None:
                skipped += 1
                continue
            if active_only and q.status.lower() != "active":
                skipped += 1
                continue
            yielded += 1
            yield q

        wb.close()
        logger.info(
            "Excel read complete — yielded: %d, skipped: %d", yielded, skipped
        )

    def load_all(self, active_only: bool = True) -> List[Question]:
        """Load all questions into memory as a list."""
        questions = list(self.iter_questions(active_only=active_only))
        logger.info("Loaded %d questions from Excel.", len(questions))
        return questions

    def get_summary(self) -> dict:
        """Quick statistics without full load."""
        questions = self.load_all(active_only=False)
        roles, types, levels = {}, {}, {}
        for q in questions:
            roles[q.role] = roles.get(q.role, 0) + 1
            types[q.question_type] = types.get(q.question_type, 0) + 1
            levels[q.experience_level] = levels.get(q.experience_level, 0) + 1
        return {
            "total": len(questions),
            "active": sum(1 for q in questions if q.status.lower() == "active"),
            "by_role": roles,
            "by_type": types,
            "by_level": levels,
        }
