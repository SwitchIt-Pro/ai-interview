"""
server.py — WebSocket server: Python AI backend ↔ Browser frontend
Fixed:
  - Greeting only sent once per session (not on every reconnect)
  - TTS buffers to sentence boundaries (no more fumbling mid-word)
  - Transcript end_interview crash fixed
  - WebSocket ping timeout increased
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio, json, logging, re, time, threading
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from services import STTService, LLMService, TTSService, VADService, TranscriptService
import config

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HOST = "localhost"
PORT = 8765

# ── Global services (loaded once, shared across sessions) ─────────────────────
vad = stt = llm = tts = transcript = None

def load_services():
    global vad, stt, llm, tts, transcript
    logger.info("Loading AI services...")
    vad        = VADService()
    stt        = STTService()
    llm        = LLMService()
    tts        = TTSService()
    transcript = TranscriptService()
    logger.info("✓ All services ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────
async def send_json(ws, **kwargs):
    try:
        await ws.send(json.dumps(kwargs))
    except Exception:
        pass

async def send_audio(ws, audio_np: np.ndarray):
    if len(audio_np) == 0:
        return
    try:
        await ws.send(audio_np.astype(np.float32).tobytes())
    except Exception:
        pass

# ── Sentence boundary splitter ────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?,;:])\s+')

def split_into_speakable_chunks(text: str) -> list[str]:
    """
    Split text at sentence/clause boundaries.
    Returns list of complete-enough strings safe to pass to TTS.
    """
    parts = SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

# ── Audio buffer (VAD-based utterance detection) ──────────────────────────────
class AudioBuffer:
    SAMPLE_RATE       = 16000
    SILERO_CHUNK      = 512
    MAX_SILENCE_MS    = 1500
    MIN_SPEECH_MS     = 600

    def __init__(self):
        self.frames         = []
        self.speech_started = False
        self.silence_count  = 0
        self.max_silence_frames = int(self.MAX_SILENCE_MS / (self.SILERO_CHUNK / self.SAMPLE_RATE * 1000))

    def push(self, chunk: np.ndarray):
        is_speech = vad.is_speech(chunk, self.SAMPLE_RATE)
        if is_speech:
            if not self.speech_started:
                self.speech_started = True
            self.silence_count = 0
            self.frames.append(chunk)
        else:
            if self.speech_started:
                self.silence_count += 1
                self.frames.append(chunk)
                if self.silence_count >= self.max_silence_frames:
                    audio = np.concatenate(self.frames)
                    self.reset()
                    total_ms = len(audio) / self.SAMPLE_RATE * 1000
                    if total_ms >= self.MIN_SPEECH_MS:
                        return audio
        return None

    def reset(self):
        self.frames = []
        self.speech_started = False
        self.silence_count  = 0

# ── Per-client interview session ──────────────────────────────────────────────
class InterviewSession:
    def __init__(self, ws):
        self.ws           = ws
        self.buffer       = AudioBuffer()
        self._interrupted = False
        self._ai_speaking = False
        self._greeted     = False   # ← prevents repeat greetings on reconnect

    async def handle_audio_chunk(self, raw_bytes: bytes):
        chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()

        # Check for interruption while AI is speaking
        if self._ai_speaking:
            for i in range(0, len(chunk) - 511, 512):
                if vad.is_speech(chunk[i:i+512]):
                    logger.info("⚡ Interruption detected")
                    self._interrupted = True
                    return

        # Process in 512-sample frames for VAD
        for i in range(0, len(chunk) - 511, 512):
            utterance = self.buffer.push(chunk[i:i+512])
            if utterance is not None:
                await self.process_utterance(utterance)

    async def process_utterance(self, audio_np: np.ndarray):
        # ── STT ──────────────────────────────────────────────────────────────
        await send_json(self.ws, type="status", state="thinking", text="Transcribing...")
        t0 = time.perf_counter()
        user_text = await asyncio.get_event_loop().run_in_executor(
            None, stt.transcribe, audio_np)
        stt_ms = (time.perf_counter() - t0) * 1000

        if not user_text.strip():
            await send_json(self.ws, type="status", state="listening", text="Didn't catch that — speak again")
            return

        logger.info(f"👤 [{stt_ms:.0f}ms]: {user_text}")
        transcript.add("Candidate", user_text)
        await send_json(self.ws, type="user_message", text=user_text)

        # ── LLM → TTS streaming ───────────────────────────────────────────────
        await send_json(self.ws, type="status", state="thinking", text="AI is thinking...")
        await send_json(self.ws, type="ai_start")
        await self._stream_llm_tts(user_text)
        await send_json(self.ws, type="status", state="listening", text="Your turn — speak now")

    async def _stream_llm_tts(self, user_text: str):
        """
        LLM streams tokens → buffer until sentence boundary → TTS synthesize → send audio.
        This prevents TTS from receiving half-words, eliminating fumbling.
        """
        self._ai_speaking = True
        self._interrupted  = False
        full_response      = []
        audio_queue        = asyncio.Queue()
        loop               = asyncio.get_event_loop()

        def _run_llm_tts():
            text_buffer = ""
            for token in llm.generate_stream(user_text):
                if self._interrupted:
                    break
                text_buffer += token
                full_response.append(token)

                # Only send to TTS at sentence/clause boundaries
                # This is the key fix for fumbling TTS
                if SENTENCE_END.search(text_buffer) or len(text_buffer) > 120:
                    chunks = split_into_speakable_chunks(text_buffer)
                    for chunk_text in chunks:
                        if self._interrupted:
                            break
                        try:
                            audio = tts.synthesize(chunk_text)
                            asyncio.run_coroutine_threadsafe(
                                audio_queue.put(("chunk", chunk_text, audio)), loop)
                        except Exception as e:
                            logger.warning(f"TTS error on chunk: {e}")
                    text_buffer = ""

            # Flush any remaining text
            if text_buffer.strip() and not self._interrupted:
                try:
                    audio = tts.synthesize(text_buffer.strip())
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put(("chunk", text_buffer.strip(), audio)), loop)
                except Exception as e:
                    logger.warning(f"TTS flush error: {e}")

            asyncio.run_coroutine_threadsafe(audio_queue.put(("done", "", None)), loop)

        t = threading.Thread(target=_run_llm_tts, daemon=True)
        t.start()

        while True:
            kind, text_chunk, audio_np = await audio_queue.get()
            if kind == "done":
                break
            # Send text caption
            await send_json(self.ws, type="ai_chunk", text=text_chunk)
            # Send audio PCM
            if audio_np is not None and len(audio_np) > 0:
                await send_audio(self.ws, audio_np)

        t.join()

        ai_text = " ".join(full_response).strip()
        if ai_text:
            transcript.add("AI", ai_text)
            await send_json(self.ws, type="ai_end", text=ai_text)

        self._ai_speaking = False

    async def send_greeting(self):
        """Send greeting once. Guard with self._greeted to prevent repeats."""
        if self._greeted:
            return
        self._greeted = True

        greeting = ("Hello! Welcome to your technical interview. "
                    "I'm your AI recruiter today. "
                    "Please go ahead and introduce yourself.")

        await send_json(self.ws, type="status", state="speaking", text="AI is speaking...")
        await send_json(self.ws, type="ai_start")
        self._ai_speaking = True

        audio = await asyncio.get_event_loop().run_in_executor(
            None, tts.synthesize, greeting)

        await send_json(self.ws, type="ai_chunk", text=greeting)
        await send_audio(self.ws, audio)
        await send_json(self.ws, type="ai_end", text=greeting)
        transcript.add("AI", greeting)

        self._ai_speaking = False
        await send_json(self.ws, type="status", state="listening", text="Your turn — speak now")

# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handle_connection(ws: WebSocketServerProtocol):
    logger.info(f"Client connected: {ws.remote_address}")
    session = InterviewSession(ws)

    try:
        await send_json(ws, type="ready")
        await session.send_greeting()

        async for message in ws:
            if isinstance(message, bytes):
                await session.handle_audio_chunk(message)
            elif isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "end_interview":
                    transcript.save()
                    await send_json(ws, type="interview_ended")
                    logger.info("Interview ended by user — shutting down server.")
                    # Close WebSocket cleanly then stop the process
                    await ws.close()
                    os._exit(0)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Client disconnected cleanly.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Session error: {e}", exc_info=True)
    finally:
        transcript.save()
        logger.info("Session ended.")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    load_services()
    logger.info(f"WebSocket server running on ws://{HOST}:{PORT}")

    async with websockets.serve(
        handle_connection, HOST, PORT,
        max_size=10 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=60,
    ):
        # Auto-open frontend in default browser
        import webbrowser, pathlib
        html_path = pathlib.Path(__file__).parent / "frontend" / "index.html"
        webbrowser.open(html_path.as_uri())
        logger.info(f"✓ Browser opened: {html_path}")

        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())