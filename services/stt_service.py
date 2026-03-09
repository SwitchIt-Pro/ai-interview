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
        """
        Sets up the Speech-to-Text (STT) service.
        Think of this as setting up a court reporter whose job is to listen to the audio and write down everything that is said verbatim.
        """
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the 'Whisper' AI model into the computer's memory.
        This is like giving our reporter a very good dictionary and training on how words sound.
        We load a smaller, optimized version (float16) to make sure it doesn't take up too much space on the graphics card (VRAM).
        """
        logger.info("Loading STT model: whisper-small (float16)...")
        self.model = WhisperModel(
            "small",                        # ~500MB VRAM
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,  # float16
        )
        logger.info("✓ STT model loaded.")

    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        Takes a piece of audio and turns it into text.
        Returns the written transcript as a string.
        
        This function also has built-in "hallucination filters." Sometimes the AI "hears" things 
        that aren't there (like thanking an imaginary audience when it's just silence). 
        The filters block these common mistakes and ignore recordings that are too short to be real speech.
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
        """
        Slowly reveals the transcribed text piece by piece (as a generator), just like live captions.
        Instead of waiting for the entire audio to finish processing, it hands over each new word or phrase as soon as it figures it out.
        """
        segments, _ = self.model.transcribe(
            audio_np,
            language=config.STT_LANGUAGE,
            beam_size=3,
        )
        for segment in segments:
            yield segment.text.strip()
