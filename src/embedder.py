"""
src/embedder.py
---------------
Generates text embeddings using OpenAI text-embedding-3-small — with PARALLEL requests.

MODEL:
  text-embedding-3-small  →  embeddings ONLY (fast, cost-effective, used for ChromaDB indexing)

SPEED:
  Uses ThreadPoolExecutor for parallel embedding calls.
  With 8 parallel workers: typically 1-3 minutes for 3,000+ questions.

Usage:
  embedder = Embedder()
  vectors  = embedder.encode(["question 1", "question 2"])
  vector   = embedder.encode_one("single question")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class Embedder:
    """
    Generates embeddings using OpenAI text-embedding-3-small — with parallel requests.

    Parallel embedding: instead of embedding one text at a time (slow),
    we send `workers` requests simultaneously to the OpenAI API. This gives
    a significant speedup for large batches.

    Usage
    -----
    embedder = Embedder(workers=8)           # default
    vectors  = embedder.encode(texts)        # parallel batch
    vector   = embedder.encode_one("text")   # single (no threads)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        workers: Optional[int] = None,
    ):
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env.\n"
                "Add: OPENAI_API_KEY=sk-proj-..."
            )
        from openai import OpenAI
        self._client   = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model    = model or config.EMBEDDING_MODEL
        self._workers  = workers if workers is not None else config.EMBEDDING_WORKERS

        logger.info(
            "Embedder ready — model: %s, workers: %d",
            self._model, self._workers,
        )

    # ── Public API ────────────────────────────────────────────────

    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Encode a list of texts into embedding vectors using OpenAI.

        Uses a ThreadPoolExecutor to send `workers` concurrent requests,
        dramatically reducing total embedding time.

        Order is preserved — embeddings[i] corresponds to texts[i].

        Parameters
        ----------
        texts         : list of strings to embed
        show_progress : print a progress counter while running

        Returns
        -------
        List of float vectors, one per input text, in the same order.
        """
        if not texts:
            return []

        total = len(texts)

        # For tiny batches, skip thread overhead and go sequential
        if total <= 2 or self._workers <= 1:
            return self._encode_sequential(texts, show_progress)

        return self._encode_parallel(texts, show_progress)

    def encode_one(self, text: str) -> List[float]:
        """Encode a single string. No threading overhead."""
        return self._embed_single(text)

    # ── Parallel Encoding ─────────────────────────────────────────

    def _encode_parallel(
        self,
        texts: List[str],
        show_progress: bool,
    ) -> List[List[float]]:
        """
        Send embedding requests to OpenAI concurrently via ThreadPoolExecutor.

        We keep a results dict indexed by position so order is preserved,
        since futures complete in non-deterministic order.
        """
        total = len(texts)
        results: dict[int, List[float]] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            # submit all tasks, tagged with position index
            future_to_idx = {
                pool.submit(self._embed_single, text): i
                for i, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Embedding failed for index %d: %s", idx, e)
                    raise RuntimeError(
                        f"OpenAI embedding failed for text at index {idx}: {e}"
                    )
                completed += 1
                if show_progress and total > 5 and completed % 5 == 0:
                    pct = completed / total * 100
                    print(
                        f"  [Embedder] {completed}/{total} ({pct:.0f}%)...",
                        end="\r",
                        flush=True,
                    )

        if show_progress and total > 5:
            print(f"  [Embedder] {total}/{total} (100%) ✓           ")

        # Reconstruct in original order
        return [results[i] for i in range(total)]

    # ── Sequential Fallback ───────────────────────────────────────

    def _encode_sequential(
        self,
        texts: List[str],
        show_progress: bool,
    ) -> List[List[float]]:
        """Sequential (one-by-one) encoding — used for tiny batches."""
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            embeddings.append(self._embed_single(text))
            if show_progress and total > 5 and (i + 1) % 5 == 0:
                print(
                    f"  [Embedder] {i + 1}/{total}...",
                    end="\r",
                    flush=True,
                )
        if show_progress and total > 5:
            print(f"  [Embedder] {total}/{total} ✓           ")
        return embeddings

    # ── Single Embedding ──────────────────────────────────────────

    def _embed_single(self, text: str) -> List[float]:
        """
        Call OpenAI Embeddings API for one text.
        Thread-safe — each call uses the shared client (which is thread-safe).
        Retries up to 3 times on transient errors.
        """
        from openai import RateLimitError, APIConnectionError, APIStatusError

        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=text,
                )
                return response.data[0].embedding

            except RateLimitError as e:
                last_exc = e
                delay = 2.0 * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI rate limit on embedding (attempt %d/%d). Retrying in %.0fs...",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except APIConnectionError as e:
                last_exc = e
                delay = 2.0 * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI connection error (attempt %d/%d). Retrying in %.0fs...",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except APIStatusError as e:
                if e.status_code >= 500:
                    last_exc = e
                    time.sleep(2.0)
                else:
                    raise RuntimeError(f"OpenAI API error: {e}")

        raise RuntimeError(
            f"OpenAI embedding failed after {_MAX_RETRIES} retries.\n"
            f"Last error: {last_exc}\n"
            f"Check your OPENAI_API_KEY and network connection."
        )

    # ── Connection Check ──────────────────────────────────────────

    def check_connection(self) -> bool:
        """Verify OpenAI API is accessible by making a test embedding call."""
        try:
            self._embed_single("connection test")
            logger.info("OpenAI Embedder OK — model '%s' ready (%d workers)", self._model, self._workers)
            return True
        except Exception as e:
            raise RuntimeError(
                f"OpenAI embedding connection check failed: {e}\n"
                f"Check your OPENAI_API_KEY in .env."
            )

    @property
    def model_name(self) -> str:
        return self._model
