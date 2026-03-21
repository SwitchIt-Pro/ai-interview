# Scout AI Interviewer

A RAG-powered AI interview engine where **Qwen-7B** acts as the interviewer and **OpenAI** meta-evaluates Qwen's performance.

768 dimensions of vectors
Cosine similarity (hnsw:space: cosine)

---

## Architecture

```
Excel Questions Store
        │
        ▼
[Qwen-7B Embedder]          ← Same Qwen model generates embeddings
        │
        ▼
   ChromaDB                 ← Vector database (cosine similarity)
        │
        ▼ (RAG retrieval)
[Qwen-7B Interviewer]       ← Qwen reads from ChromaDB, selects & asks questions
        │
        ├─► Selects best question from RAG candidates
        ├─► Optionally rephrases question to fit conversation
        └─► Scores candidate response (0–10 with rationale)
                │
                ▼
[OpenAI Meta-Evaluator]     ← OpenAI evaluates Qwen's work
        │
        ├─► Was Qwen's question appropriate? (quality audit)
        └─► Was Qwen's scoring accurate? (scoring audit)
```

### Key Design Decisions

| Component | Model | Role |
|-----------|-------|------|
| **Embeddings** | `qwen2.5:7b` (Ollama) | Generates question embeddings stored in ChromaDB |
| **Interviewer** | `qwen2.5:7b` (Ollama) | Reads from ChromaDB via RAG, selects & asks questions, scores responses |
| **Meta-Evaluator** | `gpt-4o-mini` (OpenAI) | Evaluates Qwen's question quality + scoring accuracy |

**Qwen-7B is used for BOTH embeddings and interviewing** — this ensures the semantic space used for retrieval is aligned with the model doing the querying.

**OpenAI only evaluates Qwen** — it does NOT conduct the interview. It audits whether Qwen:
1. Asked appropriate, well-calibrated questions
2. Scored responses fairly and accurately

---

## Setup

### Prerequisites

```bash
# 1. Install Ollama and pull Qwen-7B
ollama pull qwen2.5:7b
ollama serve

# 2. Install Python dependencies
pip install -r requirements.txt
```

### Configuration

Edit `.env`:

```env
OPENAI_API_KEY=sk-proj-...      # For meta-evaluation only
QWEN_MODEL=qwen2.5:7b           # Qwen model (embeddings + interviewer)
OLLAMA_BASE_URL=http://localhost:11434
```

### Data Setup

Put `questions_store.xlsx` in the `data/` folder, then:

```bash
# Load questions into ChromaDB (run ONCE)
python scripts/load_data.py

# Verify the load
python scripts/inspect_db.py
```

---

## Running an Interview

```bash
# Default (Sales Executive, Mid-level)
python scripts/run_interview.py

# Custom role and areas
python scripts/run_interview.py \
    --role "Account Manager" \
    --level "Senior" \
    --areas "Objection Handling:25:2:6:yes" \
            "Communication Skills:20:2::no" \
            "Pipeline Management:20:2" \
            "Client Relationship:20:2" \
            "Resilience:15:2"

# Without OpenAI meta-evaluation (faster, no cost)
python scripts/run_interview.py --no-meta-eval
```

### Area Format

```
"AREA_NAME:WEIGHT:QUESTIONS:MIN_SCORE:NON_NEGOTIABLE"
```

| Field | Example | Description |
|-------|---------|-------------|
| `AREA_NAME` | `Objection Handling` | Evaluation area (must match Excel data) |
| `WEIGHT` | `25` | Percentage weight (all must sum to 100) |
| `QUESTIONS` | `2` | Number of questions to ask for this area |
| `MIN_SCORE` | `6` | Minimum acceptable score (leave empty for none) |
| `NON_NEGOTIABLE` | `yes` | Whether breach triggers a hard alert |

---

## Output

After each turn, you see:

```
🤖 Qwen: How do you typically handle a prospect who pushes back on pricing?
👤 You: I start by understanding the underlying concern...

📊 Qwen Score: 7.5/10  ✅  [STRONG]
   Rationale: Candidate showed structured objection handling...
   Strengths: Clear framework, empathy demonstrated
   Weaknesses: Could be more specific on ROI calculation

⏳ OpenAI is meta-evaluating Qwen's performance...
──────────────────────────────────────────────────────────────
  OPENAI META-EVALUATOR — Turn 1
──────────────────────────────────────────────────────────────
  ✅ Question Quality: APPROVED (Overall: 8.2/10 | Relevance: 9.0 | ...)
     Feedback: The question directly tests objection handling...

  ✅ Scoring Accuracy: ACCURATE (Qwen: 7.5/10 | OpenAI: 8.0/10 | Δ=+0.5)
     Assessment: Qwen's score is consistent with the response quality...
──────────────────────────────────────────────────────────────
```

Final reports saved to `reports/`:
- `session_TIMESTAMP.json` — full structured data
- `report_TIMESTAMP.txt` — recruiter-readable report

---

## Project Structure

```
scout_ai_interviewer/
├── .env                         # API keys and config
├── requirements.txt
├── data/
│   └── questions_store.xlsx     # Question database (from Excel)
├── chroma_store/                # ChromaDB persistence (auto-created)
├── reports/                     # Interview reports (auto-created)
├── src/
│   ├── config.py                # Centralized configuration
│   ├── embedder.py              # Qwen-7B embeddings via Ollama
│   ├── excel_reader.py          # Excel → Question dataclasses
│   ├── vector_store.py          # ChromaDB persistence layer
│   ├── rag_engine.py            # High-level RAG query interface
│   ├── qwen_interviewer.py      # QwenClient + QwenInterviewer
│   ├── openai_evaluator.py      # OpenAI meta-evaluator
│   ├── interview_state.py       # Session state management
│   ├── interview_runner.py      # Full interview orchestration
│   └── report_generator.py     # Report generation
└── scripts/
    ├── load_data.py             # Excel → ChromaDB loader
    ├── run_interview.py         # Main entry point
    └── inspect_db.py           # ChromaDB inspector
```
