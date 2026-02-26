"""
src/embedder.py
---------------
Generates text embeddings using nomic-embed-text via Ollama — with PARALLEL requests.

MODEL SEPARATION:
  nomic-embed-text  →  embeddings ONLY (fast, dedicated, used here)
  qwen2.5:7b        →  interviewer ONLY (question selection + scoring in qwen_interviewer.py)
  gpt-4o-mini       →  meta-evaluator ONLY (OpenAI audits Qwen in openai_evaluator.py)

SPEED:
  nomic-embed-text is a 274 MB dedicated embedding model.
  It is ~10-20x faster than Qwen-7B for this task.
  With 8 parallel workers: ~3-10 minutes for 3,691 questions.

Setup:
  ollama pull nomic-embed-text

Usage:
  embedder = Embedder()
  vectors  = embedder.encode(["question 1", "question 2"])
  vector   = embedder.encode_one("single question")
"""

from __future__ import annotations

import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class Embedder:
    """
    Generates embeddings using Qwen-7B via Ollama — with parallel requests.

    Parallel embedding: instead of embedding one text at a time (slow),
    we send `workers` requests simultaneously to Ollama. This gives
    3–5× speedup on CPU, more on GPU.

    Usage
    -----
    embedder = Embedder(workers=4)          # default — safe for most machines
    embedder = Embedder(workers=8)          # if you have a GPU
    vectors  = embedder.encode(texts)       # parallel batch
    vector   = embedder.encode_one("text")  # single (no threads)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        workers: Optional[int] = None,
    ):
        # Uses EMBEDDING_MODEL (nomic-embed-text), NOT QWEN_MODEL.
        # Qwen remains strictly the interviewer in qwen_interviewer.py.
        self._model    = model    or config.EMBEDDING_MODEL
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._endpoint = f"{self._base_url}/api/embeddings"
        self._workers  = workers if workers is not None else config.EMBEDDING_WORKERS

        logger.info(
            "Embedder ready — model: %s, workers: %d, endpoint: %s",
            self._model, self._workers, self._endpoint,
        )

    # ── Public API ────────────────────────────────────────────────

    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Encode a list of texts into embedding vectors using Qwen-7B.

        Uses a ThreadPoolExecutor to send `workers` concurrent requests
        to Ollama, dramatically reducing total embedding time.

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
        Send embedding requests to Ollama concurrently via ThreadPoolExecutor.

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
                        f"Qwen embedding failed for text at index {idx}: {e}"
                    )
                completed += 1
                if show_progress and total > 5 and completed % 5 == 0:
                    pct = completed / total * 100
                    print(
                        f"  [Qwen Embedder] {completed}/{total} ({pct:.0f}%)...",
                        end="\r",
                        flush=True,
                    )

        if show_progress and total > 5:
            print(f"  [Qwen Embedder] {total}/{total} (100%) ✓           ")

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
                    f"  [Qwen Embedder] {i + 1}/{total}...",
                    end="\r",
                    flush=True,
                )
        if show_progress and total > 5:
            print(f"  [Qwen Embedder] {total}/{total} ✓           ")
        return embeddings

    # ── Single Embedding ──────────────────────────────────────────

    def _embed_single(self, text: str) -> List[float]:
        """
        POST to Ollama /api/embeddings for one text using Qwen-7B.
        Thread-safe — each call creates its own requests session.
        Retries up to 3 times on transient errors.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self._endpoint,
                    json={"model": self._model, "prompt": text},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                if "embedding" not in data:
                    raise RuntimeError(
                        f"Ollama/Qwen response missing 'embedding' key. Got: {data}"
                    )
                return data["embedding"]

            except requests.exceptions.ConnectionError as e:
                last_exc = e
                delay = 2.0 * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama connection failed (attempt %d/%d). Retrying in %.0fs...",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning(
                    "Qwen embedding timed out (attempt %d/%d).", attempt, _MAX_RETRIES
                )
                time.sleep(2.0)

            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"Ollama HTTP error: {e}\n"
                    f"Check if '{self._model}' is pulled: 'ollama pull {self._model}'"
                )

        raise RuntimeError(
            f"Qwen embedding failed after {_MAX_RETRIES} retries.\n"
            f"Last error: {last_exc}\n"
            f"Make sure Ollama is running: 'ollama serve'"
        )

    # ── Connection Check ──────────────────────────────────────────

    def check_connection(self) -> bool:
        """Verify Ollama is running and nomic-embed-text model is available."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if not any(self._model in m for m in models):
                raise RuntimeError(
                    f"Embedding model '{self._model}' not found in Ollama.\n"
                    f"Available: {models}\n"
                    f"Pull it: 'ollama pull {self._model}'"
                )
            logger.info("Ollama OK — embedding model '%s' ready (%d workers)", self._model, self._workers)
            return True
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama not running at {self._base_url}.\n"
                f"Start it: 'ollama serve'"
            )

    @property
    def model_name(self) -> str:
        return self._model
