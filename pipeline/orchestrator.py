"""
pipeline/orchestrator.py
The main interview pipeline. Connects all services in a streaming loop.

Flow:
  Mic → VAD → STT → LLM (stream) → TTS (stream) → Speakers
  
Key features:
  - Streaming LLM→TTS: TTS starts speaking after first ~5 LLM words
  - Interruption handling: user speech stops TTS immediately
  - Turn-taking: VAD controls who is "speaking"
  - Multi-turn memory: full conversation history passed to LLM
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import time
from services import (
    STTService, LLMService, TTSService,
    VADService, AudioCaptureService, TranscriptService
)
import config

logger = logging.getLogger(__name__)


class InterviewOrchestrator:
    def __init__(self, event_callback=None):
        logger.info("Initializing services (this may take ~30s on first run)...")
        self.vad        = VADService()
        self.stt        = STTService()
        self.llm        = LLMService()
        self.tts        = TTSService()
        self.audio      = AudioCaptureService()
        self.transcript = TranscriptService()

        self.event_callback = event_callback
        self._is_ai_speaking = False
        self._interrupted     = False
        self._is_running      = False
        logger.info("✓ All services initialized.")

    async def _emit(self, event_type: str, data: dict = None):
        if self.event_callback:
            payload = {"type": event_type}
            if data:
                payload.update(data)
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(payload)
            else:
                self.event_callback(payload)

    async def start(self):
        """Main interview loop."""
        self.audio.start()
        self._is_running = True

        logger.info("\n" + "="*50)
        logger.info("  INTERVIEW STARTED — Press Ctrl+C to stop")
        logger.info("="*50 + "\n")

        await self._emit("status", {"status": "speaking", "text": "AI is speaking..."})

        try:
            # AI speaks first (greeting)
            await self._ai_speak_stream(
                iter(["Hello! Welcome to your technical interview. I'm your AI recruiter today. "
                      "Please go ahead and introduce yourself."])
            )

            # Main conversation loop
            while self._is_running:
                await self._listen_and_respond()

        except KeyboardInterrupt:
            logger.info("\nInterview ended by user.")
        finally:
            self.stop()
            
    def stop(self):
        """Stops the interview loop."""
        self._is_running = False
        self.audio.stop()
        self.transcript.save()
        logger.info("Interview complete. Transcript saved.")

    async def _listen_and_respond(self):
        """One full turn: listen → transcribe → generate → speak."""

        # ── LISTEN ────────────────────────────────────────────────────────────
        await self._emit("status", {"status": "listening", "text": "Your turn — press mic to speak (Auto-detecting...)"})
        logger.info("🎤 Listening...")
        audio_segment = await asyncio.get_event_loop().run_in_executor(
            None,
            self.audio.collect_speech_segment,
            self.vad,
        )

        if len(audio_segment) == 0:
            return

        # Validate: enough real speech (avoid hallucination on noise)
        if not self.vad.has_enough_speech(audio_segment):
            logger.debug("Skipped: not enough speech detected.")
            return

        # ── TRANSCRIBE ────────────────────────────────────────────────────────
        await self._emit("status", {"status": "thinking", "text": "Transcribing..."})
        t0 = time.perf_counter()
        user_text = self.stt.transcribe(audio_segment)
        stt_ms = (time.perf_counter() - t0) * 1000

        if not user_text.strip():
            logger.debug("STT returned empty string.")
            return

        logger.info(f"👤 Candidate [{stt_ms:.0f}ms]: {user_text}")
        self.transcript.add("Candidate", user_text)
        await self._emit("transcript", {"role": "user", "text": user_text})

        # ── GENERATE + SPEAK (streaming) ──────────────────────────────────────
        await self._emit("status", {"status": "thinking", "text": "AI is thinking..."})
        t1 = time.perf_counter()
        llm_stream = self.llm.generate_stream(user_text)
        await self._ai_speak_stream(llm_stream, start_time=t1)

    async def _ai_speak_stream(self, text_chunk_generator, start_time: float = None):
        """
        Stream LLM chunks directly to TTS.
        TTS starts playing after the first few words — masking LLM latency.
        Monitors mic for interruptions while speaking.
        """
        self._is_ai_speaking = True
        self._interrupted     = False
        full_text = []

        await self._emit("status", {"status": "speaking", "text": "AI is speaking..."})

        def speak_in_background():
            for chunk in text_chunk_generator:
                if self._interrupted:
                    self.tts.stop()
                    break
                full_text.append(chunk)
                if start_time and len(full_text) == 1:
                    ttfs = (time.perf_counter() - start_time) * 1000
                    logger.debug(f"Time-to-first-speech: {ttfs:.0f}ms")
                # Play each chunk as it arrives (streaming TTS)
                audio = self.tts.synthesize(chunk)
                import sounddevice as sd
                import numpy as np
                if len(audio) > 0:
                    sd.play(audio, samplerate=config.TTS_SAMPLE_RATE)
                    sd.wait()

        # Run TTS in executor (non-blocking) so we can watch for interruptions
        loop = asyncio.get_event_loop()
        speak_task = loop.run_in_executor(None, speak_in_background)

        # Watch for interruptions while AI is speaking
        while not speak_task.done():
            chunk = await loop.run_in_executor(
                None, self.audio.read_chunk, 0.05
            )
            if chunk is not None and self.vad.is_speech(chunk):
                logger.info("⚡ Interruption detected — stopping AI speech.")
                self._interrupted = True
                self.tts.stop()
                break
            await asyncio.sleep(0.02)

        await speak_task

        ai_text = " ".join(full_text).strip()
        if ai_text:
            logger.info(f"🤖 AI: {ai_text}")
            self.transcript.add("AI", ai_text)
            await self._emit("transcript", {"role": "ai", "text": ai_text})

        self._is_ai_speaking = False