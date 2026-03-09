# Conversational AI Scout — RTX 1650 Edition

Voice-to-voice AI interviewer optimized for **RTX 1650 (4GB VRAM)**.

---

## VRAM Budget
| Component | Model | VRAM |
|-----------|-------|------|
| STT | Whisper Small (float16) | ~500MB |
| LLM | Qwen2.5-1.5B Q4 | ~1.2GB |
| TTS | Kokoro-82M | ~300MB |
| OS + Overhead | — | ~2.0GB |
| **Total** | | **~4.0GB ✓** |

---

## Project Structure
```
conversational-ai-scouts/
├── main.py                  # Entry point
├── config.py                # All settings (tweak here)
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py      # Core streaming loop
├── services/
│   ├── __init__.py
│   ├── audio_service.py     # Mic capture
│   ├── vad_service.py       # Silero VAD (CPU)
│   ├── stt_service.py       # Whisper Small (CUDA)
│   ├── llm_service.py       # Qwen2.5-1.5B 4-bit (CUDA)
│   └── tts_service.py       # Kokoro-82M (CUDA)
├── logs/                    # Auto-created, interview transcripts saved here
└── frontend/                # Reserved for future Web UI
```

---

## Setup

### 1. Prerequisites
- Windows 11 / Ubuntu 22.04+
- Python 3.11
- CUDA 12.1+ drivers ([download](https://developer.nvidia.com/cuda-downloads))
- RTX 1650 with latest NVIDIA driver (545+)

### 2. Install PyTorch (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install project dependencies
```bash
pip install -r requirements.txt
```

### 4. First run (models auto-download ~2GB total)
```bash
python main.py
```
Models are cached in `~/.cache/huggingface/` after first download.

---

## Running
```bash
python main.py
```
- Speak naturally into your mic
- AI will respond via speakers
- Press **Ctrl+C** to end the interview
- Transcript is auto-saved to `logs/interview_YYYYMMDD_HHMMSS.txt`

---

## Tuning for Your Setup

| Setting | File | What it does |
|---------|------|--------------|
| `VAD_SILENCE_MS` | config.py | How long to wait after you stop talking |
| `LLM_MAX_NEW_TOKENS` | config.py | Longer = more detailed AI answers |
| `TTS_VOICE` | config.py | `af_sarah`, `am_adam`, `bf_emma` |
| `SYSTEM_PROMPT` | config.py | Change interview style/persona |

---

## Why Different from the PDF

The PDF targets RTX 3090/4090 (24GB). For RTX 1650 (4GB):

| PDF Spec | This Project | Reason |
|----------|-------------|--------|
| Canary Qwen 2.5B | Whisper Small | 2.5B STT needs 3GB+ alone |
| Qwen3-7B Instruct | Qwen2.5-1.5B Q4 | 7B needs 8GB+ even quantized |
| CosyVoice 0.5B | Kokoro-82M | Kokoro is lighter, easier install |
| LiveKit + Pipecat | Direct sounddevice | Full WebRTC overkill for local use |

---

## Troubleshooting

**"CUDA out of memory"**
```python
# In config.py, reduce:
LLM_MAX_NEW_TOKENS = 100   # was 200
```

**Models downloading slowly**
- First run only. Subsequent runs use cache.

**No audio input/output**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
# Find your device index, then in audio_service.py add: device=YOUR_INDEX
```

**bitsandbytes not working on Windows**
```bash
pip install bitsandbytes --prefer-binary
# Or use: https://github.com/jllllll/bitsandbytes-windows-webui
```
