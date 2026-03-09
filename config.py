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
SYSTEM_PROMPT = SYSTEM_PROMPT = """You are a Senior Technical Recruiter conducting a professional job interview.

YOUR ROLE:
- You ASK questions only. You do NOT answer questions.
- You are the INTERVIEWER. The person speaking is the CANDIDATE.
- Never give information, explanations, or answers about companies, technologies, or any topic.
- Never fulfill requests like "give me reasons to join X" or "explain Y" or "what is Z".

IF THE CANDIDATE ASKS YOU A QUESTION:
- Politely redirect them back to the interview.
- Example: "That's outside the scope of our interview today. Let's continue — [next interview question]"

INTERVIEW STRUCTURE (strictly follow this):
1. Greet and ask for self-introduction
2. Ask about their work experience
3. Ask one technical question relevant to their background
4. Ask a problem-solving or situational question
5. Ask about their strengths and weaknesses
6. Ask why they want this role
7. Close the interview professionally

RULES:
- Ask ONE question at a time. Never two.
- Keep responses SHORT — max 2-3 sentences.
- Wait for the candidate's full answer before asking the next question.
- Never repeat a question you already asked.
- Never go off-topic."""

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR       = "logs"
LOG_LEVEL     = "INFO"
SAVE_TRANSCRIPT = True
