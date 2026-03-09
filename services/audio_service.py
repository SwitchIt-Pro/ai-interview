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
        """
        Sets up the Audio Capture service.
        Think of this as getting a microphone ready to record audio. We prepare a waiting line (queue) 
        where small pieces of sound will wait to be processed, and get our internal states ready.
        """
        self._audio_queue = queue.Queue()
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        """
        This is a hidden helper function called automatically every time the microphone captures a small piece of sound.
        It simply takes that sound piece and puts it in our waiting line (queue) so we can process it later.
        """
        if status:
            logger.warning(f"Audio input status: {status}")
        self._audio_queue.put(indata[:, 0].copy())  # Mono

    def start(self):
        """
        Turns the microphone on.
        It starts continuously listening and passing small, fast snippets of sound to our helper function to be saved.
        """
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
        """
        Turns the microphone off.
        It stops listening and safely closes the connection to the computer's audio system.
        """
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        logger.info("Microphone stopped.")

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Grabs the next piece of sound from our waiting line (queue).
        If the microphone hasn't picked up anything new within a short time limit (timeout), it gives up and returns nothing.
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def collect_speech_segment(self, vad_service, max_duration_sec: float = 30.0) -> np.ndarray:
        """
        Listens to the microphone continuously and gathers all the sound into one big recording, but only when someone is actually talking.
        
        How it works:
        - It ignores silence until someone starts speaking.
        - Once speech starts, it records everything.
        - If the person stops talking and there's silence for a little bit, it assumes they're done and stops.
        - It cuts off automatically if the recording gets too long (e.g., 30 seconds).
        Returns the glued-together speech audio.
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
