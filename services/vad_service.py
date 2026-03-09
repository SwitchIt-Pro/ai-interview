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
        """
        Sets up the Voice Activity Detection (VAD) service.
        Think of this as preparing a listener whose only job is to hear if someone is currently speaking.
        We start with empty placeholders for the model and utility tools.
        """
        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self):
        """
        Loads the 'Silero VAD' AI model into the computer's standard memory (CPU).
        This is like giving our listener their brain and training so they know what human speech sounds like.
        We run it on CPU so it doesn't take up the limited space on the graphics card.
        """
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
        Listens to a small slice of audio and guesses if it contains human speech.
        
        The model is very picky and insists on hearing exactly 512 tiny audio dots (samples) at a time.
        If we give it a snippet that's too small, we pad it with silence.
        If we give it a snippet that's too large, we chop it into chunks, ask the model about each piece,
        and average out its answers to give a final 'Yes' or 'No'.
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
        """
        Looks at a longer piece of audio and figures out the exact start and end times (timestamps) 
        of when someone was talking.
        
        Like a timeline editor, it slices the audio, ignores small gaps of silence, and tells us 
        the specific windows where actual speech happened.
        """
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
        """
        Checks if the person spoke for a long enough time (by default at least 300 milliseconds).
        
        Sometimes there's just a tiny noise like a cough or a mic bump. This function uses the timestamps 
        from the previous method to add up all the speaking time. If the total speaking time is 
        longer than our minimum limit, it returns True, meaning "Yes, that was a real sentence or word."
        """
        timestamps = self.get_speech_timestamps(audio_np, sample_rate)
        if not timestamps:
            return False
        total_speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        total_speech_ms = (total_speech_samples / sample_rate) * 1000
        return total_speech_ms >= min_duration_ms