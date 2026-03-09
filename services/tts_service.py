"""
services/tts_service.py
Text-to-Speech using Kokoro-82M
Optimized for RTX 1650 — ~300MB VRAM, ~120ms latency
Kokoro is lightweight, fast, and sounds natural. Much lighter than CosyVoice 0.5B.
"""

import torch
import numpy as np
import logging
import sounddevice as sd
from kokoro import KPipeline
import config

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load Kokoro TTS pipeline."""
        logger.info(f"Loading TTS: Kokoro-82M (voice={config.TTS_VOICE})...")
        # 'a' = American English, 'b' = British English
        lang_code = "a"
        self.pipeline = KPipeline(lang_code=lang_code)
        logger.info("✓ TTS model loaded.")

    def synthesize_stream(self, text_chunks):
        """
        Accept a generator of text chunks, synthesize each chunk, play audio.
        This runs in lock-step with the LLM streamer — 
        starts speaking before LLM finishes generating.
        """
        audio_buffer = []

        for text_chunk in text_chunks:
            if not text_chunk.strip():
                continue

            logger.debug(f"TTS synthesizing chunk: '{text_chunk[:40]}...'")

            # Generate audio for this chunk
            generator = self.pipeline(
                text_chunk,
                voice=config.TTS_VOICE,
                speed=config.TTS_SPEED,
                split_pattern=None,     # Disable auto-split, we handle chunks
            )

            for _, _, audio in generator:
                audio_np = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
                audio_buffer.append(audio_np)
                self._play_audio(audio_np)

        return np.concatenate(audio_buffer) if audio_buffer else np.array([])

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize full text and return numpy audio array."""
        audio_chunks = []
        generator = self.pipeline(
            text,
            voice=config.TTS_VOICE,
            speed=config.TTS_SPEED,
        )
        for _, _, audio in generator:
            audio_np = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
            audio_chunks.append(audio_np)

        return np.concatenate(audio_chunks) if audio_chunks else np.array([])

    def _play_audio(self, audio_np: np.ndarray):
        """Play audio array through default speakers (blocking per-chunk)."""
        sd.play(audio_np, samplerate=config.TTS_SAMPLE_RATE)
        sd.wait()

    def stop(self):
        """Interrupt current playback (called on user interruption)."""
        sd.stop()
        logger.debug("TTS playback interrupted.")
