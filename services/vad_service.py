"""
services/vad_service.py
Voice Activity Detection using Silero VAD
CPU-based — saves precious VRAM for LLM/STT/TTS

IMPORTANT: Silero VAD requires EXACTLY 512 samples at 16000Hz per call.
Chunks larger than 512 are split and averaged.
"""

import torch
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)

SILERO_CHUNK = 512   # Exactly required by Silero at 16kHz


class VADService:
    def __init__(self):
        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self):
        logger.info("Loading VAD: Silero VAD (CPU)...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.eval()
        logger.info("✓ VAD model loaded on CPU.")

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Returns True if speech is detected.
        Silero requires EXACTLY 512 samples — we split larger chunks and average.
        """
        audio_chunk = audio_chunk.astype(np.float32)

        total = len(audio_chunk)
        if total < SILERO_CHUNK:
            audio_chunk = np.pad(audio_chunk, (0, SILERO_CHUNK - total))
            total = SILERO_CHUNK

        probs = []
        for i in range(0, total - SILERO_CHUNK + 1, SILERO_CHUNK):
            frame = audio_chunk[i:i + SILERO_CHUNK]
            tensor = torch.from_numpy(frame).float().unsqueeze(0)  # (1, 512)
            with torch.no_grad():
                prob = self.model(tensor, sample_rate).item()
            probs.append(prob)

        avg_prob = sum(probs) / len(probs) if probs else 0.0
        return avg_prob >= config.VAD_THRESHOLD

    def get_speech_timestamps(self, audio_np: np.ndarray, sample_rate: int = 16000):
        get_speech_ts = self.utils[0]
        tensor = torch.from_numpy(audio_np.astype(np.float32))
        timestamps = get_speech_ts(
            tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=config.VAD_THRESHOLD,
            min_silence_duration_ms=config.VAD_SILENCE_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        )
        return timestamps

    def has_enough_speech(self, audio_np: np.ndarray, sample_rate: int = 16000, min_duration_ms: int = 300) -> bool:
        timestamps = self.get_speech_timestamps(audio_np, sample_rate)
        if not timestamps:
            return False
        total_speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        total_speech_ms = (total_speech_samples / sample_rate) * 1000
        return total_speech_ms >= min_duration_ms