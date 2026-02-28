# Scout AI Interviewer

A RAG-powered AI interview engine built entirely on **OpenAI** — embeddings, interviewing, scoring, meta-evaluation, and candidate simulation all use the OpenAI API. No local models required.

---

## Architecture

```
Excel Questions Store
        │
        ▼
[OpenAI Embedder]           ← text-embedding-3-small generates question embeddings
        │
        ▼
   ChromaDB                 ← Vector database (cosine similarity, persistent)
        │
        ▼ (RAG retrieval)
[OpenAI Interviewer]        ← gpt-4o-mini reads from ChromaDB, selects & asks questions
        │
        ├─► Selects best question from RAG candidates
        ├─► Optionally rephrases question to fit conversation tone
        └─► Scores candidate response (0–10 with rationale)
                │
                ▼
[OpenAI Simulator]          ← gpt-4o-mini simulates candidate responses (default mode)
                │
                ▼
[OpenAI Meta-Evaluator]     ← gpt-4o-mini audits interview quality (optional, --meta-eval)
        │
        ├─► Was the question appropriate? (quality audit)
        └─► Was the scoring accurate? (scoring audit)
```

### Components

| Component | Model | Role |
|-----------|-------|------|
| **Embeddings** | `text-embedding-3-small` (OpenAI) | Embeds questions into ChromaDB + embeds RAG queries |
| **Interviewer** | `gpt-4o-mini` (OpenAI) | Selects questions via RAG, rephrases, scores responses |
| **Simulator** | `gpt-4o-mini` (OpenAI) | Auto-generates candidate answers (default mode) |
| **Meta-Evaluator** | `gpt-4o-mini` (OpenAI) | Audits question quality + scoring accuracy (optional) |

**Single API key** — everything uses `OPENAI_API_KEY`. No local models, no Ollama.

---

## Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

Edit `.env`:

```env
OPENAI_API_KEY=sk-proj-...           # Required — all roles use this key

EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_WORKERS=8

OPENAI_INTERVIEWER_MODEL=gpt-4o-mini
OPENAI_INTERVIEWER_TEMPERATURE=0.7

OPENAI_EVAL_MODEL=gpt-4o-mini
OPENAI_EVAL_TEMPERATURE=0.2
```

### Data Setup

Put `questions_store.xlsx` in the `data/` folder, then load it into ChromaDB:

```bash
# First time (or after changing embedding model)
python scripts/load_data.py

# Force full reload (wipes existing ChromaDB)
python scripts/load_data.py --force-reload

# Incremental sync (only add new / remove deleted questions)
python scripts/load_data.py --sync
```

> ⚠️ If you previously used a different embedding model, always use `--force-reload` to avoid dimension mismatch errors.

---

## Running an Interview

```bash
# Default — OpenAI simulates candidate answers, no meta-evaluation
python scripts/run_interview.py

# Type your own answers (interactive mode)
python scripts/run_interview.py --interactive

# Choose candidate persona for simulation
python scripts/run_interview.py --persona strong    # strong / average / weak

# Enable OpenAI meta-evaluation after each turn
python scripts/run_interview.py --meta-eval

# Custom role and level
python scripts/run_interview.py --role "Account Manager" --level "Senior"

# Custom evaluation areas
python scripts/run_interview.py \
    --role "Account Manager" \
    --level "Senior" \
    --areas "Objection Handling:25:2:6:yes" \
             "Communication Skills:20:2::no" \
             "Pipeline Management:20:2" \
             "Client Relationship:20:2" \
             "Resilience:15:2"

# Save JSON + text reports to reports/
python scripts/run_interview.py --export
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--role` | `Sales Executive` | Job role |
| `--level` | `Mid-level` | `Fresher` / `Early Career` / `Mid-level` / `Senior` |
| `--areas` | Default Sales config | Custom evaluation areas (see format below) |
| `--persona` | `average` | Simulation persona: `strong` / `average` / `weak` |
| `--interactive` | Off | You type answers instead of OpenAI simulating |
| `--meta-eval` | Off | Enable OpenAI quality + scoring audit after each turn |
| `--export` | Off | Save JSON + text report to `reports/` |
| `--verbose` | Off | Enable debug logging |

### Area Format

```
"AREA_NAME:WEIGHT:QUESTIONS:MIN_SCORE:NON_NEGOTIABLE"
```

| Field | Example | Description |
|-------|---------|-------------|
| `AREA_NAME` | `Objection Handling` | Evaluation area (must match Excel) |
| `WEIGHT` | `25` | Percentage weight (all must sum to 100) |
| `QUESTIONS` | `2` | Number of questions to ask |
| `MIN_SCORE` | `6` | Minimum acceptable score (leave empty for none) |
| `NON_NEGOTIABLE` | `yes` | Whether breach triggers a hard alert |

---

## Output

After each turn:

```
🤖 AI: How do you typically handle a prospect who pushes back on pricing?

🎭 Simulated [STRONG]: I start by understanding the underlying concern...

⏳ AI is scoring the response...
📊 AI Score: 8.5/10  ✅  [STRONG]
   Rationale: Candidate showed a structured approach to objection handling...
   Strengths: Clear framework, empathy demonstrated, ROI-focused
   Weaknesses: Could include more specific metrics

═════════════════════════════════════════════════════════════════
  LIVE SCORECARD (AI Interview Score)
═════════════════════════════════════════════════════════════════
  Objection Handling                    : 8.5/10 (28.90/34%)
-----------------------------------------------------------------
  Overall Score                         : 28.90/100
═════════════════════════════════════════════════════════════════
```

Final summary printed at end of session. Use `--export` to save reports to `reports/`:
- `session_TIMESTAMP.json` — full structured data
- `report_TIMESTAMP.txt` — human-readable recruiter report

---

## Project Structure

```
scout_ai_interviewer_openai/
├── .env                          # API keys and config (not committed)
├── requirements.txt
├── data/
│   └── questions_store.xlsx      # Question database (from Excel)
├── chroma_store/                 # ChromaDB persistence (auto-created)
├── reports/                      # Interview reports (auto-created)
├── src/
│   ├── config.py                 # Centralized configuration
│   ├── embedder.py               # OpenAI text-embedding-3-small
│   ├── excel_reader.py           # Excel → Question dataclasses
│   ├── vector_store.py           # ChromaDB persistence layer
│   ├── rag_engine.py             # High-level RAG query interface
│   ├── openai_interviewer.py     # OpenAIClient + OpenAIInterviewer
│   ├── openai_evaluator.py       # OpenAI meta-evaluator (optional)
│   ├── candidate_simulator.py    # OpenAI candidate simulator
│   ├── interview_state.py        # Session state management
│   ├── interview_runner.py       # Full interview orchestration
│   └── report_generator.py      # Report generation
└── scripts/
    ├── load_data.py              # Excel → ChromaDB loader
    ├── run_interview.py          # Main entry point
    └── inspect_db.py            # ChromaDB inspector
```

---

## Requirements

```
chromadb>=0.5.0
openpyxl>=3.1.2
python-dotenv>=1.0.0
openai>=1.12.0
rich>=13.7.0
```

---

## Windows Note

If you see Unicode errors on Windows, prefix commands with:

```powershell
$env:PYTHONUTF8=1; python scripts/run_interview.py
```

Or set it permanently:

```powershell
[System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```
