"""
services/audio_service.py
Microphone capture using sounddevice.
Captures audio in chunks, applies VAD, returns speech segments.
No WebRTC needed for local-only deployment on RTX 1650.
"""

import numpy as np
import sounddevice as sd
import logging
import queue
import threading

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16000   # Hz — required by Whisper and Silero VAD
CHUNK_DURATION_MS = 96  # ms per chunk
CHUNK_SIZE    = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # = 480 samples
MAX_SILENCE_CHUNKS = int(700 / CHUNK_DURATION_MS)          # ~700ms silence


class AudioCaptureService:
    def __init__(self):
        self._audio_queue = queue.Queue()
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            logger.warning(f"Audio input status: {status}")
        self._audio_queue.put(indata[:, 0].copy())  # Mono

    def start(self):
        """Start microphone capture stream."""
        logger.info(f"Starting mic capture (SR={SAMPLE_RATE}Hz, chunk={CHUNK_DURATION_MS}ms)...")
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("✓ Microphone active.")

    def stop(self):
        """Stop microphone capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        logger.info("Microphone stopped.")

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read one audio chunk from the queue. Returns None on timeout."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def collect_speech_segment(self, vad_service, max_duration_sec: float = 30.0) -> np.ndarray:
        """
        Collect audio until VAD detects end of speech (silence).
        Returns the full speech segment as a float32 numpy array.
        
        Logic:
          - Wait for speech to START (skip leading silence)
          - Collect audio while speech is active
          - Stop after MAX_SILENCE_CHUNKS of consecutive silence
        """
        speech_started = False
        silence_count   = 0
        frames = []

        max_chunks = int(max_duration_sec * 1000 / CHUNK_DURATION_MS)
        collected = 0

        logger.debug("Waiting for speech...")

        while collected < max_chunks:
            chunk = self.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            is_speech = vad_service.is_speech(chunk, sample_rate=SAMPLE_RATE)

            if is_speech:
                if not speech_started:
                    logger.debug("Speech detected — recording...")
                    speech_started = True
                silence_count = 0
                frames.append(chunk)
            else:
                if speech_started:
                    silence_count += 1
                    frames.append(chunk)  # Include trailing silence for natural cut
                    if silence_count >= MAX_SILENCE_CHUNKS:
                        logger.debug(f"End of speech detected after {len(frames)} chunks.")
                        break

            collected += 1

        if not frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(frames)
