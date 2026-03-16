# Conversational AI Scout — RTX 1650 Edition

> A fully local, voice-to-voice AI interviewer for SwitchIt-Pro. No internet. No cloud. Runs on your own PC.

---

## 🧠 What Does This Project Do? (Plain English)

This project creates an **AI-powered interviewer** that can talk to a job candidate completely on its own.

Here's the simple idea:
1. A candidate sits down and **speaks into a microphone**
2. The AI **listens**, understands what they said, **thinks of a reply**, and **speaks back** — just like a real interviewer would
3. The whole conversation is **saved as a text log** so a recruiter can review it later

All of this happens **in real time**, on a single gaming PC (NVIDIA RTX 1650), with **no cloud subscription or internet connection needed**.

---

## 🗣️ What Kind of Interviews Does It Conduct?

The AI acts as a **friendly, professional Technical Interviewer** for a Round 1 screening call at SwitchIt-Pro. It follows this structure every time:

| Step | What Happens |
|------|--------------|
| 1. Warm Introduction | The AI greets the candidate and asks them to introduce themselves |
| 2. Resume / Experience | Asks about recent work or a project they are proud of |
| 3. Basic Technical Question | Tests a simple concept related to the job |
| 4. Intermediate Technical | Asks how they'd tackle a common technical problem |
| 5. Advanced Technical | Digs into deeper topics like system design or edge cases |
| 6. Professional Closing | Asks if the candidate has questions, then wraps up warmly |

The AI asks **one question at a time**, waits for the full answer, uses friendly phrases like *"That makes sense!"*, and keeps responses short (2–3 sentences max).

---

## ⚙️ How Does It Work? (The Pipeline — Simply Explained)

Think of the system as an assembly line with 6 workers:

```
[You speak] → [Listener] → [Transcriber] → [AI Brain] → [Voice Box] → [Recorder]
```

| Worker | Tech Used | Job |
|--------|-----------|-----|
| 🎙️ Audio Service | sounddevice | Picks up your voice from the microphone |
| 👂 VAD Service | Silero VAD | Decides when you START and STOP talking (ignores background noise) |
| 📝 STT Service | Whisper Small | Converts your speech into written text |
| 🧠 LLM Service | Qwen2.5-1.5B | Reads the text and thinks up the AI's reply |
| 🔊 TTS Service | Kokoro-82M | Turns the AI's words into a human-sounding voice and plays it |
| 📂 Transcript Service | Python file I/O | Quietly saves the entire conversation to a text file |

---

## 💾 Why Does It Run on a Budget GPU?

The project was carefully designed to run on a **4 GB VRAM** GPU (RTX 1650) by picking smaller, optimised versions of each AI model:

| What It Does | Model Chosen | Memory Used |
|--------------|-------------|-------------|
| Hear your speech | Whisper Small | ~500 MB |
| Think & reply | Qwen2.5-1.5B (4-bit compressed) | ~1.2 GB |
| Speak back to you | Kokoro-82M | ~300 MB |
| Windows + Python overhead | — | ~2.0 GB |
| **Total** | | **~4.0 GB ✓** |

---

## 📁 What's in Each File?

| File / Folder | What It Is |
|---------------|-----------|
| `main.py` | The **on/off switch** — starts the server and waits for a frontend to connect |
| `config.py` | The **settings panel** — change voices, AI behaviour, and thresholds here |
| `requirements.txt` | The **shopping list** of all Python libraries needed |
| `pipeline/orchestrator.py` | The **conductor** — coordinates all services from start to finish |
| `services/audio_service.py` | The **ears** — captures mic audio |
| `services/vad_service.py` | The **attention filter** — knows when you're actually talking |
| `services/stt_service.py` | The **court reporter** — types out what you said |
| `services/llm_service.py` | The **brain** — generates the AI's response |
| `services/tts_service.py` | The **vocal cords** — speaks the AI's response out loud |
| `services/transcript_service.py` | The **filing cabinet** — logs every word spoken |
| `logs/` | Where completed interview transcripts are saved automatically |
| `frontend/` | Reserved for a future web interface (not built yet) |

---

## 🚀 How to Run It

### Step 1 — Install PyTorch (GPU version)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Install everything else
```bash
pip install -r requirements.txt
```

### Step 3 — Start the server
```bash
python main.py
```
> ℹ️ The first run will download AI models (~2 GB total). After that, they're cached locally and it starts instantly.

---

## 🎛️ Things You Can Tweak (in `config.py`)

| Setting | What It Changes |
|---------|----------------|
| `VAD_SILENCE_MS` | How long of silence before the AI decides you've finished talking |
| `LLM_MAX_NEW_TOKENS` | How long the AI's answers can be |
| `TTS_VOICE` | The voice character (`af_sarah`, `am_adam`, `bf_emma`) |
| `SYSTEM_PROMPT` | The AI's personality and interview script |

---

## 🛠️ Common Problems

**"CUDA out of memory"**
→ Open `config.py` and lower `LLM_MAX_NEW_TOKENS` from `200` to `100`

**No mic / speaker audio**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
→ Find your device index and set it in `audio_service.py`

**Models downloading slowly**
→ This only happens on the very first run. After that, they load from your local cache.

---

## 📄 Output

Every interview session is automatically saved to:
```
logs/interview_YYYYMMDD_HHMMSS.txt
```
This file contains a timestamped back-and-forth log of everything the candidate and AI said.

---

*SwitchIt-Pro · conversational-ai-scouts · RTX 1650 Edition*
