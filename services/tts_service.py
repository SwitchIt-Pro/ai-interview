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
        """
        Sets up the Text-to-Speech (TTS) service.
        Think of this as giving our AI interviewer a vocal cord so it can actually speak out loud.
        """
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """
        Loads the voice generation model (Kokoro) into memory.
        This model is chosen because it runs very fast and sounds human-like, 
        even on a basic graphics card, without taking extra space.
        """
        logger.info(f"Loading TTS: Kokoro-82M (voice={config.TTS_VOICE})...")
        # 'a' = American English, 'b' = British English
        lang_code = "a"
        self.pipeline = KPipeline(lang_code=lang_code)
        logger.info("✓ TTS model loaded.")

    def synthesize_stream(self, text_chunks):
        """
        Takes an incoming stream of text pieces (as they are being thought up by the AI brain) 
        and instantly turns them into actual sound (audio).
        
        It plays the sound through the speakers immediately, keeping the conversation fast and natural 
        without awkward pauses waiting for the full sentence to finish typing.
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
        """
        Takes a full, complete piece of text and turns it into one big chunk of audio data.
        Unlike the stream version, this waits until the whole sentence is generated before returning the sound file.
        """
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
        """
        A helper function that sends the constructed audio data straight to the computer's speakers to be played out loud.
        It waits until the short audio clip is fully played before moving on.
        """
        sd.play(audio_np, samplerate=config.TTS_SAMPLE_RATE)
        sd.wait()

    def stop(self):
        """
        Immediately stops any audio that is currently playing out loud.
        Useful for when the user accidentally interrupts the AI and we need the AI to stop talking quickly.
        """
        sd.stop()
        logger.debug("TTS playback interrupted.")
