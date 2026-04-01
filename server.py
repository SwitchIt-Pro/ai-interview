import sys, os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asyncio, json, logging, re, time, threading, uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from services import STTService, TTSService, VADService, TranscriptService, LLMService
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vad, stt, llm, tts, transcript
    logger.info("Loading AI services...")
    vad        = VADService()
    stt        = STTService()
    llm        = LLMService()   
    tts        = TTSService()
    transcript = TranscriptService()
    logger.info("✓ All services ready.")
    yield
    global _active_ws_count
    if _active_ws_count > 0:
        logger.warning(
            f"Server shutting down with {_active_ws_count} active interview session(s). "
            "Saving transcript..."
        )
    if transcript:
        try:
            transcript.save()
        except Exception:
            pass
    logger.info("Shutdown complete.")

app = FastAPI(title="Scout AI Interviewer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global services ──
vad = stt = llm = tts = transcript = None

# ── Active WebSocket session counter (protects in-flight interviews) ──
_active_ws_count = 0

# (Lifespan handles startup/shutdown)

# Ensure frontend directory is mounted for static files
# We mount it at the root
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_portal():
    return FileResponse(
        os.path.join(frontend_dir, "portal.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

class ApplyRequest(BaseModel):
    role: str
    level: str
    industry: str
    resume_text: str
    jd_text: str

# In-memory session store
sessions = {}
_prepare_executor = ThreadPoolExecutor(max_workers=2)

def _background_prepare(session_id: str, request: ApplyRequest):
    """Runs LLM prepare_session in a background thread so /api/apply returns immediately."""
    try:
        logger.info(f"[{session_id}] Background preparation started...")
        llm.prepare_session(
            session_id=session_id,
            role=request.role,
            level=request.level,
            industry=request.industry,
            resume_text=request.resume_text,
            jd_text=request.jd_text
        )
        llm_session = llm.sessions.get(session_id, {})
        is_eligible = llm_session.get("is_eligible", True)
        if not is_eligible:
            sessions[session_id]["status"] = "ineligible"
            sessions[session_id]["reason"] = llm_session.get("ineligibility_reason", "Experience level or domain does not match the JD.")
        else:
            sessions[session_id]["status"] = "ready"
        logger.info(f"[{session_id}] ✓ Background preparation complete.")
    except Exception as e:
        sessions[session_id]["status"] = "error"
        logger.error(f"[{session_id}] Background preparation failed: {e}", exc_info=True)

@app.post("/api/apply")
def api_apply(request: ApplyRequest):
    session_id = str(uuid.uuid4())
    logger.info(f"API Apply called (non-blocking): {request.role} ({request.level}) -> {session_id}")

    # Register session immediately so WebSocket won't reject it
    sessions[session_id] = {
        "role": request.role,
        "level": request.level,
        "status": "preparing"   # will flip to 'ready' when background thread finishes
    }

    # Kick off LLM/VectorDB prep in background — do NOT await it here
    _prepare_executor.submit(_background_prepare, session_id, request)

    return {"status": "success", "session_id": session_id}

@app.get("/api/session_status/{session_id}")
def api_session_status(session_id: str):
    """Lightweight poll endpoint the frontend uses to know when the LLM is ready."""
    sess = sessions.get(session_id)
    if not sess:
        return {"status": "not_found"}
    return {"status": sess.get("status", "preparing"), "reason": sess.get("reason", "")}

# ── Sentence boundary splitter ──
SENTENCE_END = re.compile(r'(?<=[.!?,;:])\s+')

def split_into_speakable_chunks(text: str) -> list[str]:
    parts = SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

class AudioBuffer:
    SAMPLE_RATE = 16000
    SILERO_CHUNK = 512
    MAX_SILENCE_MS = 1500
    MIN_SPEECH_MS = 600

    def __init__(self):
        self.frames = []
        self.speech_started = False
        self.silence_count = 0
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
        self.silence_count = 0

class InterviewSession:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.ws = websocket
        self.session_id = session_id
        self.buffer = AudioBuffer()
        self._interrupted = False
        self._ai_speaking = False
        self._greeted = False
        self._answer_start: float = 0.0   # timestamp when AI finished speaking

    async def send_json(self, **kwargs):
        try:
            await self.ws.send_json(kwargs)
        except Exception:
            pass

    async def send_audio(self, audio_np: np.ndarray):
        if len(audio_np) == 0:
            return
        try:
            await self.ws.send_bytes(audio_np.astype(np.float32).tobytes())
        except Exception:
            pass

    async def handle_audio_chunk(self, raw_bytes: bytes):
        chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()

        if self._ai_speaking:
            for i in range(0, len(chunk) - 511, 512):
                if vad.is_speech(chunk[i:i+512]):
                    logger.info("⚡ Interruption detected")
                    self._interrupted = True
                    return

        for i in range(0, len(chunk) - 511, 512):
            utterance = self.buffer.push(chunk[i:i+512])
            if utterance is not None:
                await self.process_utterance(utterance)

    async def process_utterance(self, audio_np: np.ndarray):
        await self.send_json(type="status", state="thinking", text="Transcribing...")
        t0 = time.perf_counter()
        user_text = await asyncio.get_event_loop().run_in_executor(None, stt.transcribe, audio_np)
        stt_ms = (time.perf_counter() - t0) * 1000

        if not user_text.strip():
            await self.send_json(type="status", state="listening", text="Didn't catch that — speak again")
            return

        # Measure response delay (time from when AI stopped speaking to first word)
        response_delay = time.perf_counter() - self._answer_start if self._answer_start else 0.0

        logger.info(f"🧑‍💼 [{stt_ms:.0f}ms | delay={response_delay:.1f}s]: {user_text}")
        transcript.add("Candidate", user_text)
        await self.send_json(type="user_message", text=user_text)

        await self.send_json(type="status", state="thinking", text="AI is thinking...")
        await self.send_json(type="ai_start")
        await self._stream_llm_tts(user_text, response_delay)
        if getattr(self, "_banned", False):
            await self.send_json(type="interview_ended")
            transcript.save()
            try:
                await self.ws.close(code=1008, reason="Banned")
            except Exception:
                pass
            return
        await self.send_json(type="status", state="listening", text="Your turn — speak now")

    async def _stream_llm_tts(self, user_text: str, timing: float = 0.0):
        self._ai_speaking = True
        self._interrupted = False
        self._answer_start = 0.0
        full_response = []
        audio_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_llm_tts():
            text_buffer = ""
            for token_or_chunk, score in llm.generate_stream(self.session_id, user_text, timing):
                if self._interrupted:
                    break

                # (None, None) is the keepalive sentinel yielded just before the
                # blocking Ollama scoring call — send a status ping immediately so
                # the browser knows the server is alive and doesn't drop the WS.
                if token_or_chunk is None and score is None:
                    asyncio.run_coroutine_threadsafe(
                        self.send_json(type="status", state="thinking", text="Scoring your answer..."),
                        loop
                    )
                    continue

                # Send score update to frontend (adaptive difficulty gauge)
                if score is not None:
                    asyncio.run_coroutine_threadsafe(
                        self.send_json(type="difficulty_update", value=round(score / 10.0, 2)), loop
                    )

                if token_or_chunk == "__BAN__":
                    self._banned = True
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put(("ban", "", None)), loop
                    )
                    break
                elif token_or_chunk:
                    text_buffer += token_or_chunk
                    full_response.append(token_or_chunk)

                    if SENTENCE_END.search(text_buffer) or len(text_buffer) > 120:
                        chunks = split_into_speakable_chunks(text_buffer)
                        for chunk_text in chunks:
                            if self._interrupted:
                                break
                            try:
                                audio = tts.synthesize(chunk_text)
                                asyncio.run_coroutine_threadsafe(
                                    audio_queue.put(("chunk", chunk_text, audio)), loop
                                )
                            except Exception as e:
                                logger.warning(f"TTS error: {e}")
                        text_buffer = ""

            if text_buffer.strip() and not self._interrupted:
                try:
                    audio = tts.synthesize(text_buffer.strip())
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put(("chunk", text_buffer.strip(), audio)), loop
                    )
                except Exception:
                    pass

            asyncio.run_coroutine_threadsafe(audio_queue.put(("done", "", None)), loop)

        t = threading.Thread(target=_run_llm_tts, daemon=True)
        t.start()

        # Drain audio queue — send a keepalive ping every 3 s while the LLM is
        # scoring / generating so the browser never sees an idle WebSocket and
        # drops the connection.
        KEEPALIVE_INTERVAL = 3.0  # seconds
        while True:
            try:
                kind, text_chunk, audio_np = await asyncio.wait_for(
                    audio_queue.get(), timeout=KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                # Nothing from LLM yet — send a heartbeat so the browser stays connected
                await self.send_json(type="status", state="thinking", text="AI is thinking...")
                continue

            if kind == "done" or kind == "ban":
                break
            await self.send_json(type="ai_chunk", text=text_chunk)
            if audio_np is not None and len(audio_np) > 0:
                await self.send_audio(audio_np)

        t.join()

        ai_text = " ".join(full_response).strip()
        if ai_text:
            transcript.add("AI", ai_text)
            await self.send_json(type="ai_end", text=ai_text)

        self._ai_speaking = False
        # Start timing how long until the candidate responds
        self._answer_start = time.perf_counter()

    async def send_greeting(self):
        if self._greeted:
            return
        self._greeted = True

        greeting = llm.get_opening_line(self.session_id)
        
        await self.send_json(type="status", state="speaking", text="AI is speaking...")
        await self.send_json(type="ai_start")
        self._ai_speaking = True

        audio = await asyncio.get_event_loop().run_in_executor(None, tts.synthesize, greeting)

        await self.send_json(type="ai_chunk", text=greeting)
        await self.send_audio(audio)
        await self.send_json(type="ai_end", text=greeting)
        transcript.add("AI", greeting)

        self._ai_speaking = False
        await self.send_json(type="status", state="listening", text="Your turn — speak now")


@app.websocket("/ws/interview")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    global _active_ws_count
    await websocket.accept()
    if session_id not in sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    _active_ws_count += 1
    session = InterviewSession(websocket, session_id)
    _connected = True
    try:
        await session.send_json(type="ready")

        while _connected:
            try:
                message = await websocket.receive()
            except RuntimeError as exc:
                # Starlette raises RuntimeError("Cannot call 'receive' once a
                # disconnect message has been received.") — treat as clean disconnect.
                if "disconnect" in str(exc).lower():
                    logger.info("Client disconnected (RuntimeError guard).")
                else:
                    logger.error(f"WebSocket receive error: {exc}", exc_info=True)
                break

            # Starlette signals disconnection via a special message type
            if message.get("type") == "websocket.disconnect":
                logger.info("Client disconnected (disconnect frame).")
                _connected = False
                break

            if "bytes" in message:
                await session.handle_audio_chunk(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "start_interview":
                    await session.send_greeting()
                elif data.get("type") == "end_interview":
                    transcript.save()
                    await session.send_json(type="interview_ended")
                    # Send Phase 3 report to frontend if ready
                    report = llm.get_report(session_id)
                    if report:
                        await session.send_json(
                            type="phase3_report",
                            role_fit_pct=report.get("role_fit_pct"),
                            decision=report.get("decision"),
                            scores=report.get("scores"),
                        )
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    _connected = False
                    break
    except WebSocketDisconnect:
        logger.info("Client disconnected (WebSocketDisconnect).")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        _active_ws_count -= 1
        transcript.save()


@app.get("/api/report/{session_id}")
def api_get_report(session_id: str):
    """Poll this after interview ends to get the Phase 3 HR report JSON."""
    report = llm.get_report(session_id)
    if not report:
        return {"status": "pending"}
    return {"status": "ready", **report}

if __name__ == "__main__":
    import webbrowser
    html_url = "http://localhost:8000/"
    webbrowser.open(html_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    