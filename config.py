"""
config.py - Central configuration

VRAM Budget (~4GB):
  - STT  (Whisper Small):     ~0.5GB
  - TTS  (Kokoro-82M):        ~0.3GB
  - OS/Overhead:              ~2.0GB
  --------------------------------
  Total:                      ~2.8GB ✓
  (LLM runs via Ollama external to this VRAM budget)
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

# ─── LLM: Qwen2.5 via Ollama (Zero VRAM impact for this script) ───────────────
OLLAMA_BASE_URL    = "http://localhost:11434"
LLM_MODEL          = "qwen2.5:1.5b"
LLM_MAX_NEW_TOKENS = 350
LLM_TEMPERATURE    = 0.7

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
SYSTEM_PROMPT = """You are a friendly Technical Interviewer conducting a Round 1 introductory screening interview for a software engineer role at SwitchIt. The interview should take about 10-15 minutes.

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
- Never go off-topic.
- VERY IMPORTANT — DO NOT HALLUCINATE: If you did not clearly understand what the candidate said, or if their answer is unclear or incomplete, ask a short clarifying question such as "Could you elaborate on that a bit more?" or "Sorry, could you repeat that?" — NEVER make up or assume what they said.
- If the candidate mentions a fact, skill, or experience you are not sure about, do NOT invent details. Instead, ask a follow-up question to learn more from them directly.
- Stay strictly within the interview flow. If something seems off or unclear, ask the candidate to clarify."""

# ─── RAG (Retrieval-Augmented Generation) ────────────────────────────────────
# Controls how the conversational AI queries the scout_ai_interviewer vector DB.
# At interview start the LLM fetches a structured question plan from ChromaDB:
#   Q1 = intro, Q2–Q(N-1) = role-specific domain questions, QN = outro
RAG_ENABLED          = True          # Set False to disable RAG entirely

# Total number of questions to ask per interview (min 3, max 10).
# Breakdown:  1 intro  +  (INTERVIEW_NUM_QUESTIONS - 2) domain  +  1 outro
INTERVIEW_NUM_QUESTIONS = 6          # 1 intro + 3 domain + 1 outro

# Candidate context — set these to match the interview being conducted.
# These filters are sent to ChromaDB so only relevant questions are retrieved.
# Common roles: "Software Engineer", "Sales Executive", "Data Scientist", etc.
RAG_CANDIDATE_ROLE     = "Software Engineer"

# Industry filter — set to None to search across all industries.
# Common values: "Technology", "Finance", "Healthcare", "Retail", None
RAG_CANDIDATE_INDUSTRY = None          # e.g. "Technology" or None for all

# Experience level filter — set to None to search across all levels.
# Common values: "Junior", "Mid-level", "Senior", None
RAG_CANDIDATE_LEVEL    = None          # e.g. "Mid-level" or None for all levels

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR       = "logs"
LOG_LEVEL     = "INFO"
SAVE_TRANSCRIPT = True
