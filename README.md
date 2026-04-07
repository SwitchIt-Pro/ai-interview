# Scout AI Interviewer — Question Bank Manager

The **Scout AI Interviewer** is the question generation and VectorDB management layer for the SwitchIt-Pro platform. It provides a clean, recruiter-facing web portal to generate, store, and manage role-specific interview questions — fully locally, using Ollama + Qwen.

Questions generated here are consumed directly by the **Conversational AI Scouts** pipeline, which uses them to conduct live voice-based screening interviews.

---

## How It Works

```
Recruiter enters Role + Level
          ↓
Qwen2.5:1.5b synthesizes JD context (if none provided)
          ↓
Qwen extracts skills + evaluation areas
          ↓
Questions generated in batches of 10 (fits 1.5b context window)
          ↓
Each question embedded via nomic-embed-text → ChromaDB (scout_questions)
          ↓
Raw data saved to data/questions.xlsx (8-column schema)
          ↓
VectorDB updated ✅ — live interviews can now use these questions
```

---

## 4-Page Secure Web Portal

The portal is secured by a login overlay. Only administrators with valid credentials in the `.env` file can access the web application.

### Page 1 — Question Bank (Home)
- Grid of all generated roles in the VectorDB
- Shows **role name**, **experience level badge**, **skills extracted**, and **question count**
- **Search** by role name + filter by experience level
- **Check VectorDB** button — confirms ChromaDB connectivity and total question count
- **Refresh** button to reload from server

### Page 2 — Add Questions
- Enter **Role Name** + **Experience Level** — that's all that's required
- JD is **optional** — if left blank, Qwen auto-generates a contextualized job description
- Questions generated in **batches of 10** to fit the 1.5b model's context window
- Live **progress bar** + **log console** showing every step
- Result panel shows questions added + skills Qwen extracted

### Page 3 — Upload Sheet
- Directly upload `.xlsx` or `.xls` files with exactly the **8-column schema**.
- **Bypasses LLM generation**: Instantly embeds rows and syncs to ChromaDB.
- **Smart Validation**: Blocks invalid schemas with clear error feedback.
- **Deduplication**: Automatically skips questions that already exist in the database.
- Shows live animation progress and instantly reports rows added and skipped.

### Page 4 — Manage Roles
- Full table of all roles with search + level filter
- **Delete** button per role — removes from both `questions.xlsx` and ChromaDB atomically via confirmation modal

---

## Question Schema (8 Columns)

Every generated question is stored with exactly these fields:

| Column | Description |
|--------|-------------|
| `question_id` | Auto-assigned (`Q0001`, `Q0002`…) |
| `question_text` | The actual interview question |
| `role` | Exact role name entered |
| `evaluation_area` | One of 5 criteria (see below) |
| `experience_level` | Fresher / Mid-level / Senior |
| `question_type` | Behavioral / Situational / Profile-Grounded / Follow-up / Knowledge Check |
| `what_ai_listens_for` | 1–2 concrete sentences on scoring signals |
| `follow_up_trigger` | Short phrase (<10 words) that should trigger a follow-up probe |

**Evaluation Areas (exactly 5):**
- `Communication Skills`
- `Conceptual Clarity`
- `Problem Solving`
- `Role Relevance`
- `Resume Authenticity`

**Experience Levels (exactly 3):**
- `Fresher` — 0–2 years
- `Mid-level` — 3–10 years
- `Senior` — 10+ years

---

## VG ID (Variant Group ID)

Every generation run produces a **Variant Group ID** (`VG0001`, `VG0002`…). It groups all questions belonging to the same role + level batch. Used for:
- Counting questions per role card on the home page
- Deleting an entire role's questions in one click
- Duplicate JD detection — warns if a similar role already exists

VG IDs are stored in ChromaDB's `jd_registry` collection (not in Excel).

---

## Prerequisites

```bash
# 1. Ollama must be running
ollama serve

# 2. Pull required models
ollama pull qwen2.5:1.5b      # LLM for question generation
ollama pull nomic-embed-text  # Embedding model for ChromaDB
```

---

## Setup & Run

```bash
cd scout_ai_interviewer
pip install -r requirements.txt
python server.py
```

Navigate to **`http://localhost:8050`**

---

## ChromaDB Collections

| Collection | Purpose |
|------------|---------|
| `scout_questions` | All generated interview questions with embeddings |
| `jd_registry` | One entry per VG — role, level, skills, creation date |

---

## Project Structure

```
scout_ai_interviewer/
├── server.py                 # FastAPI backend (generation, embedding, CRUD)
├── portal.html               # 3-page frontend (Vanilla JS/CSS)
├── requirements.txt
├── .env                      # Configuration overrides
├── frontend/                 # Client-side web application source
│   ├── index.html
│   └── style.css
├── data/
│   ├── questions.xlsx        # Master question store (8-column schema)
│   └── questions_store_empty.xlsx  # Blank template (reference only)
├── chroma_store/             # ChromaDB persistent storage
├── src/
│   ├── config.py             # Paths, model names, ChromaDB settings
│   ├── embedder.py           # Parallel Ollama embedding wrapper
│   ├── excel_reader.py       # Excel → Question dataclass reader
│   ├── vector_store.py       # ChromaDB upsert / search / delete
│   └── sync_service.py       # Background Excel-to-Chroma file watcher
└── scripts/
    ├── load_data.py          # Manual bulk loader (Excel → ChromaDB)
    └── inspect_db.py         # CLI inspector for ChromaDB state
```

---

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_EMAIL` | `admin@scout.ai` | Admin portal login email |
| `ADMIN_PASSWORD` | `Scout@2024` | Admin portal login password |
| `SESSION_SECRET` | `fallback-secret-change-me` | Secret used to sign session cookies |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_WORKERS` | `8` | Parallel embedding threads |
| `CHROMA_PERSIST_DIR` | `chroma_store` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `scout_questions` | Main ChromaDB collection |
| `BATCH_SIZE` | `50` | Embedding batch size |
| `SYNC_POLL_INTERVAL` | `10` | File watcher poll interval (seconds) |

---

## CLI Tools

```bash
# Manual bulk load from Excel to ChromaDB
python scripts/load_data.py
python scripts/load_data.py --sync           # Incremental only
python scripts/load_data.py --force-reload   # Wipe and reload all
python scripts/load_data.py --summary        # Stats only

# Inspect ChromaDB
python scripts/inspect_db.py                              # Collection stats
python scripts/inspect_db.py --search "objection handling"  # Semantic search
python scripts/inspect_db.py --role "Sales Executive"       # Filter by role
python scripts/inspect_db.py --id Q0001                     # Lookup by ID
```

---

## Cloud Deployment

For deploying on AWS EC2 with a GPU-enabled instance, refer to [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).
