# Conversational AI Scout — SwitchIt-Pro Edition

> A fully local, voice-to-voice AI interviewer for SwitchIt-Pro. Adaptive. Scored. No cloud required.

---

## 🧠 What Does This Project Do?

This project creates an **AI-powered interviewer** that conducts real, personalised job-screening interviews — entirely on your own machine.

Here's the flow in plain English:
1. A candidate visits the **web portal**, uploads their resume, and clicks **Apply**
2. The AI reads their resume + the job description and **generates personalised questions** in the background
3. Once ready, the candidate enters the interview room and **speaks into their microphone**
4. The AI **listens**, understands, **scores the answer**, **adapts the next question** to their performance, and **speaks back** — just like a real interviewer
5. After 7 questions the interview ends and a full **HR report + candidate feedback** is generated automatically

All of this runs **in real time, fully locally** — no OpenAI calls for the interview itself, no cloud subscription, no internet dependency for the core pipeline.

---

## 🎯 The Three-Phase Interview Pipeline

### Phase 1 — Pre-Interview Preparation (runs in background on Apply)
- **Parses** the candidate's resume and the job description using `Qwen2.5:7b` (via Ollama)
- **Generates personalised Q1 and Q2** — not generic questions, but questions referencing the candidate's actual experience and the company's specific JD
- **Searches the VectorDB** (ChromaDB with 3,691 question documents) for role-relevant question candidates

### Phase 2 — Live Adaptive Interview (7 questions)
Every answer is scored and drives the next question:

| Question | Type | Scoring Parameter | Generated From |
|----------|------|-------------------|----------------|
| Q1 | Personalised intro (from resume) | Communication | Phase 1 |
| Q2 | Personalised motivation (from JD) | Communication | Phase 1 |
| Q3 | Resume deep-dive | Resume Authenticity | Q1 answer |
| Q4 | Conceptual probe | Conceptual Clarity | Q2 answer |
| Q5 | Role relevance (VectorDB-adapted) | Role Relevance | Q3 answer |
| Q6 | Problem-solving | Problem Solving | Q4 answer |
| Q7 | Professional closing | Communication | Q5 answer |

**n-2 Adaptive Logic:** When you score Q(n), the system immediately starts generating Q(n+2) using that score — so questions arrive without delay and are dynamically calibrated:
- Weak answer → probes deeper into the gap
- Strong answer → advances to a harder angle
- Surface-level answer → requests a concrete implementation example

### Phase 3 — Post-Interview Reports (runs in background after closing)
- **5-parameter scoring** (Conceptual Clarity, Resume Authenticity, Role Relevance, Communication, Problem Solving)
- **Hidden signal capture**: response delay on technical questions, repeated filler phrases, confidence drop between early and late questions, surface-level answer pattern, resume contradictions
- **HR Decision**: Shortlist / HR Review / Reject
- **Candidate feedback**: 7-8 personalised improvement points, warm tone, no scores revealed
- Saved to `logs/` as both `.txt` transcript and structured JSON

---

## ⚙️ Architecture

```
Browser (portal.html)
    │
    ├── POST /api/apply          → Registers session, kicks off Phase 1 in background thread
    ├── GET /api/session_status  → Poll until status = "ready" (frontend blocks entry until confirmed)
    ├── WebSocket /ws/interview  → Full-duplex audio + JSON messaging
    └── GET /api/report          → Poll for Phase 3 HR report after interview ends
```

### Audio Pipeline (per candidate turn)

```
[Mic Input] → [VAD: Silero] → [Buffer] → [STT: Whisper Small]
                                                    ↓
                                          [LLM: Qwen2.5:7b via Ollama]
                                          ├── Score previous answer
                                          ├── Generate next question (n-2 adaptive)
                                          └── Stream question word-by-word
                                                    ↓
                                          [TTS: Kokoro-82M → PCM audio]
                                                    ↓
                                          [WebSocket → Browser AudioContext]
```

---

## 🛠️ Tech Stack

| Component | Model / Library | Purpose |
|-----------|----------------|---------|
| **STT** | `whisper-small` (float16, faster-whisper) | Speech → Text |
| **LLM** | `qwen2.5:7b` via Ollama | Question generation, answer scoring, adaptive logic |
| **Embeddings** | `nomic-embed-text` via Ollama | VectorDB semantic search |
| **VectorDB** | ChromaDB (`scout_questions`, 3,691 docs) | Role-specific question retrieval |
| **TTS** | `Kokoro-82M` (CPU, af_sarah voice) | Text → Natural speech |
| **VAD** | Silero VAD (CPU) | Detects speech start/end, filters silence |
| **Meta-Evaluator** | `gpt-4o-mini` (OpenAI) | Optional Phase 3 quality check |
| **Backend** | FastAPI + Uvicorn | REST API + WebSocket server |
| **Frontend** | Vanilla HTML/CSS/JS | Candidate portal in a single `portal.html` |

---

## 📁 Project Structure

```
conversational-ai-scouts/
├── server.py                   ← FastAPI app: REST API + WebSocket + graceful shutdown
├── config.py                   ← Central settings (VAD, STT, TTS, interview persona)
├── requirements.txt
├── frontend/
│   ├── portal.html             ← Full candidate portal: apply → wait → interview → report
│   └── portal.css              ← External stylesheet for the portal
├── services/
│   ├── llm_service.py          ← 3-phase adaptive pipeline (Phase 1/2/3 logic)
│   ├── stt_service.py          ← Whisper Small transcription
│   ├── tts_service.py          ← Kokoro-82M synthesis
│   ├── vad_service.py          ← Silero VAD speech detection
│   ├── transcript_service.py   ← Saves .txt + HR report + candidate feedback
│   ├── rag_service.py          ← RAG search helpers (used by llm_service)
│   └── audio_service.py        ← Mic capture (CLI pipeline, not used by server)
└── logs/
    ├── interview_YYYYMMDD_HHMMSS.txt    ← Timestamped conversation transcript
    ├── hr_report_*.json                 ← Structured HR decision + scores
    └── candidate_feedback_*.txt         ← Personalised improvement feedback
```

> The VectorDB lives at `../scout_ai_interviewer/chroma_store` (sibling directory).

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+ (Anaconda recommended)
- NVIDIA GPU with 4 GB+ VRAM **or** run on CPU (slower, but supported)
- [Ollama](https://ollama.com) installed and running

### Step 1 — Install PyTorch (GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Pull Ollama models
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```
> Make sure Ollama is running before starting the server: `ollama serve`

### Step 4 — (Optional) Set OpenAI key for Phase 3 meta-evaluation
```bash
set OPENAI_API_KEY=sk-...
```
> If not set, Phase 3 still runs but skips the OpenAI quality-check step.

### Step 5 — Start the server
```bash
python server.py
```
> This automatically opens `http://localhost:8000` in your browser.  
> First run downloads Whisper and Kokoro models (~800 MB total). After that, they load from cache.

---

## 🖥️ Using the Portal

1. **Choose a role** — Recruiter (Mid-level) or Sales Executive
2. **Upload your resume** — paste text, upload a `.pdf`, or upload a `.txt`
3. **Click Apply** — the server immediately starts preparing your personalised interview in the background
4. **Wait for the progress bar to hit 100%** — the Enter button only unlocks once the server confirms `status: ready` (usually 1–2 minutes for Q1/Q2 generation + VectorDB search)
5. **Enter the Interview Room** — your mic is initialised, the AI greets you, and the interview begins
6. **Speak naturally** — the AI listens, scores each answer, and adapts the next question
7. **Interview ends after Q7** — an HR report and candidate feedback are generated in the background

---

## 🎛️ Settings (`config.py`)

| Setting | Default | What It Changes |
|---------|---------|-----------------|
| `TTS_VOICE` | `af_sarah` | Voice character (`af_sarah`, `am_adam`, `bf_emma`) |
| `TTS_SPEED` | `1.0` | Playback speed of AI voice |
| `VAD_THRESHOLD` | `0.85` | Speech detection sensitivity (higher = stricter) |
| `VAD_SILENCE_MS` | `1500` | Silence duration before AI decides you've finished |
| `LLM_MAX_NEW_TOKENS` | `200` | Max length of AI responses |
| `INTERVIEW_NUM_QUESTIONS` | `6` (legacy) | Overridden by adaptive pipeline (7 questions) |
| `SYSTEM_PROMPT` | _(see config)_ | AI personality and interview structure |

---

## 🛠️ Common Problems

**Server shows "Ollama connection failed" on startup**
```bash
ollama serve
```
→ Ollama must be running before you start `server.py`. The embedder retries 3× and then falls back to fallback questions — the interview still works.

**"CUDA out of memory"**
→ Open `config.py` and set `STT_COMPUTE_TYPE = "int8"`. Whisper will use less VRAM.

**Enter button never becomes clickable**
→ The button only unlocks when the server responds `status: ready`. Check the terminal for `✓ Background preparation complete`. If it shows `Background preparation failed`, Ollama likely isn't running.

**No mic audio / browser permission denied**
→ Make sure you granted microphone permission when prompted. Check `chrome://settings/content/microphone` if the browser blocked it.

**AI greeting says "Hello. Let's begin your interview." instead of a personalised opening**
→ This is the fallback greeting — it means `llm.sessions` didn't have the session ready yet. Ensure preparation completed before entering (progress bar must reach 100%).

**"Server disconnected" mid-interview**
→ Check the terminal. If you see `INFO: Shutting down`, Uvicorn received a shutdown signal (e.g. closing the terminal). Restart with `python server.py` and don't close the terminal window during the interview.

---

## 📄 Output Files

| File | Contents |
|------|----------|
| `logs/interview_YYYYMMDD_HHMMSS.txt` | Full timestamped transcript — all AI and candidate turns |
| `logs/hr_report_*.json` | Role fit %, decision, per-parameter scores, red flags, strongest signal |
| `logs/candidate_feedback_*.txt` | 7-8 personalised improvement points for the candidate |

---

## 🧠 VRAM Budget (RTX 1650 — 4 GB)

| Component | Memory |
|-----------|--------|
| Whisper Small (float16) | ~500 MB |
| Kokoro-82M TTS | ~300 MB |
| Windows + Python overhead | ~2.0 GB |
| Silero VAD | ~50 MB |
| **Total GPU** | **~2.85 GB ✓** |

> `Qwen2.5:7b` runs via Ollama on CPU/RAM — it does **not** consume GPU VRAM.  
> The GPU is used exclusively for Whisper (STT) and Kokoro (TTS).

---

*SwitchIt-Pro · conversational-ai-scouts · Adaptive 3-Phase Edition*
