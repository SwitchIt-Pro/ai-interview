"""
services/stt_service.py
Speech-to-Text using Whisper Small
Optimized for RTX 1650 (4GB VRAM) — ~500MB VRAM usage
"""

import torch
import numpy as np
import logging
from faster_whisper import WhisperModel
import config

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper Small with float16 to minimize VRAM."""
        logger.info("Loading STT model: whisper-small (float16)...")
        self.model = WhisperModel(
            "small",                        # ~500MB VRAM
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,  # float16
        )
        logger.info("✓ STT model loaded.")

    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        Transcribe a numpy audio array (float32, 16kHz mono).
        Returns the transcribed text string.
        """
        if audio_np is None or len(audio_np) == 0:
            return ""

        segments, info = self.model.transcribe(
            audio_np,
            language=config.STT_LANGUAGE,
            beam_size=3,
            no_speech_threshold=0.8,
            log_prob_threshold=-0.5,
            vad_filter=False,               # VAD handled separately by Silero
        )

        text = " ".join(seg.text.strip() for seg in segments)
        text = text.strip()

        # Whisper hallucination filter
        HALLUCINATIONS = [
            "thank you", "thanks for watching", "thanks for listening",
            "please subscribe", "subtitles by", "transcribed by",
            "you", "uh", "um", "hmm", "hm", "ah", "oh",
            ".", "..", "...", "bye", "goodbye",
        ]
        if text.lower().strip(".!?, ") in HALLUCINATIONS:
            logger.debug(f"STT hallucination blocked: '{text}'")
            return ""

        # Reject very short transcriptions (1-2 words are usually noise)
        if len(text.split()) < 3:
            logger.debug(f"STT too short, likely noise: '{text}'")
            return ""

        logger.debug(f"STT result: '{text}'")
        return text

    def transcribe_stream(self, audio_np: np.ndarray):
        """Generator: yield text chunks as they are transcribed."""
        segments, _ = self.model.transcribe(
            audio_np,
            language=config.STT_LANGUAGE,
            beam_size=3,
        )
        for segment in segments:
            yield segment.text.strip()
