"""
config.py - Central configuration for RTX 1650 (4GB VRAM)

VRAM Budget (~4GB):
  - LLM  (Qwen2.5 1.5B Q4):  ~1.2GB
  - STT  (Whisper Small):     ~0.5GB
  - TTS  (Kokoro-82M):        ~0.3GB
  - OS/Overhead:              ~2.0GB
  --------------------------------
  Total:                      ~4.0GB ✓
"""

# ─── LiveKit Transport ────────────────────────────────────────────────────────
LIVEKIT_URL  = "ws://localhost:7880"     # Local LiveKit server
LIVEKIT_TOKEN = "your_token_here"        # Generate via LiveKit CLI
LIVEKIT_ROOM  = "interview-room"

# ─── STT: Whisper Small (replaces Canary Qwen 2.5B - too large for 4GB) ──────
STT_MODEL     = "openai/whisper-small"   # ~500MB VRAM
STT_DEVICE    = "cuda"
STT_LANGUAGE  = "en"
STT_COMPUTE_TYPE = "float16"             # Use fp16 to save VRAM

# ─── LLM: Qwen2.5-1.5B-Instruct Q4 (replaces 7B - too large for 4GB) ────────
LLM_MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_DEVICE    = "cuda"
LLM_MAX_NEW_TOKENS = 200
LLM_TEMPERATURE    = 0.7
LLM_LOAD_IN_4BIT   = True               # bitsandbytes 4-bit quant = ~1.2GB
LLM_STREAM         = True

# ─── TTS: Kokoro-82M (replaces CosyVoice 0.5B - simpler setup, low latency) ─
TTS_MODEL     = "hexgrad/Kokoro-82M"     # ~300MB VRAM, ~120ms latency
TTS_VOICE     = "af_sarah"              # Options: af_sarah, am_adam, bf_emma
TTS_SPEED     = 1.0
TTS_SAMPLE_RATE = 24000

# ─── VAD: Silero VAD ─────────────────────────────────────────────────────────
VAD_MODEL         = "silero_vad"
VAD_THRESHOLD     = 0.85   # Much stricter — only triggers on clear speech
VAD_SILENCE_MS    = 1500    # Wait a bit longer before cutting off
VAD_SPEECH_PAD_MS = 30     # Less padding around speech

# ─── Pipeline ─────────────────────────────────────────────────────────────────
MAX_LATENCY_MS    = 500                 # Human tolerance threshold
PIPELINE_CHUNK_MS = 20                  # Audio chunk size

# ─── Interview Persona ────────────────────────────────────────────────────────
INTERVIEW_DURATION_MIN = 10
SYSTEM_PROMPT = """You are a friendly hiring for Switch it software engineer role and welcoming Technical Interviewer conducting a Round 1 introductory screening interview. The interview should take about 10-15 minutes.
start with above line and then ask questions to get to know the candidate and their skills.
YOUR ROLE & TONE:
- Be highly encouraging, friendly, and conversational.
- You ASK questions to get to know the candidate and their skills.
- The person speaking is the CANDIDATE.
- No need to evaluate their answers or give feedback right now; just listen, acknowledge positively, and move to the next question.

INTERVIEW STRUCTURE (strictly follow this):
1. Warm Introduction: Greet the candidate enthusiastically, introduce yourself, and ask them for a brief introduction.
2. Resume/Experience: Ask a broad question about their recent work experience or a project they enjoyed.
3. Basic Technical Question: Ask a fundamental conceptual question related to their field to warm them up.
4. Intermediate Technical Question: Ask how they would approach a common technical problem or scenario.
5. Advanced Technical Question: Ask a deeper technical question about architecture, scaling, or handling edge cases.
6. Professional Closing: Ask if they have any questions for you, then wrap up the interview politely and wish them a great day.

RULES:
- Ask ONE question at a time. Never ask a multi-part question.
- Keep your responses and questions SHORT — maximum 2-3 sentences.
- Wait for the candidate's full answer before moving on.
- Acknowledge their answers with brief, friendly affirmations (e.g., "That makes sense!", "Interesting approach!") before asking the next question.
- Never go off-topic."""

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR       = "logs"
LOG_LEVEL     = "INFO"
SAVE_TRANSCRIPT = True
