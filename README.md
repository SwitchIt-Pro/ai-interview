# Scout AI — VectorDB Sync Service

A single-purpose service that keeps a **ChromaDB** vector database in sync with an Excel question store (`questions_store.xlsx`). Uses **nomic-embed-text** via **Ollama** for generating embeddings.

## What It Does

- **Watches** `questions_store.xlsx` for changes (file hash polling)
- **Adds** new rows → generates embeddings via Ollama → inserts into ChromaDB
- **Removes** deleted rows → deletes from ChromaDB
- **Skips** unchanged rows (no re-embedding needed)

## Architecture

```
questions_store.xlsx
        │
        ▼  (file hash change detected)
  ExcelQuestionReader  →  diff question IDs
        │                       │
        ▼                       ▼
  New rows → Embedder (Ollama/nomic-embed-text) → ChromaDB (upsert)
  Removed rows → ChromaDB (delete)
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** running locally:
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```

## Setup

```bash
cd scout_ai_interviewer
pip install -r requirements.txt
```

## Usage

### Sync Service (watch mode — auto-syncs on file changes)
```bash
python -m src.sync_service
```

### Single sync pass (run once and exit)
```bash
python -m src.sync_service --once
```

### Check status (Excel vs ChromaDB diff)
```bash
python -m src.sync_service --status
```

### Force reload (wipe + re-embed everything)
```bash
python -m src.sync_service --force-reload
```

### Manual load (legacy CLI)
```bash
python scripts/load_data.py               # Full load
python scripts/load_data.py --sync        # Incremental sync
python scripts/load_data.py --summary     # Excel stats
python scripts/load_data.py --delete      # Wipe ChromaDB
```

### Inspect ChromaDB
```bash
python scripts/inspect_db.py                                    # Collection stats
python scripts/inspect_db.py --search "objection handling"      # Semantic search
python scripts/inspect_db.py --role "Sales Executive"           # Filter by role
python scripts/inspect_db.py --id Q0001                         # Lookup by ID
```

## Testing

### 1. Check current status
```bash
python -m src.sync_service --status
```
Shows Excel rows vs ChromaDB documents and whether they are in sync.

### 2. Test adding rows
1. Open `data/questions_store.xlsx`
2. Add a new row at the bottom (e.g., ID: `Q9999`, fill in question text and other fields)
3. Save and close the Excel file
4. Run a single sync:
   ```bash
   python -m src.sync_service --once
   ```
5. Output should show **"Added 1"** — the new row was embedded and inserted

### 3. Test removing rows
1. Open `data/questions_store.xlsx`
2. Delete the row you just added (or any row)
3. Save and close
4. Run sync:
   ```bash
   python -m src.sync_service --once
   ```
5. Output should show **"Removed 1"** — the deleted row was removed from ChromaDB

### 4. Test watch mode (auto-sync)
1. Start the watcher:
   ```bash
   python -m src.sync_service
   ```
2. It will print "Watching for changes..."
3. Open the Excel file, add or remove a row, save it
4. Within ~10 seconds the service auto-detects the change and syncs
5. Press `Ctrl+C` to stop

### 5. Verify with the inspector
```bash
python scripts/inspect_db.py --id Q9999           # Check if the row was added
python scripts/inspect_db.py --search "teamwork"   # Semantic search
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model in Ollama |
| `EMBEDDING_WORKERS` | `8` | Parallel embedding threads |
| `CHROMA_PERSIST_DIR` | `chroma_store` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `scout_questions` | ChromaDB collection name |
| `EXCEL_FILE_PATH` | `data/questions_store.xlsx` | Excel source file |
| `BATCH_SIZE` | `50` | Embedding batch size |
| `SYNC_POLL_INTERVAL` | `10` | File change poll interval (seconds) |

## Project Structure

```
scout_ai_interviewer/
├── .env                      # Configuration
├── requirements.txt          # Python dependencies
├── data/
│   └── questions_store.xlsx  # Source Excel file
├── chroma_store/             # ChromaDB persistent storage
├── src/
│   ├── config.py             # Loads .env settings
│   ├── embedder.py           # Ollama embedding client (parallel)
│   ├── excel_reader.py       # Excel → Question dataclass reader
│   ├── vector_store.py       # ChromaDB wrapper (upsert/search/delete)
│   └── sync_service.py       # File watcher + sync logic (MAIN SERVICE)
└── scripts/
    ├── load_data.py          # Manual CLI loader
    └── inspect_db.py         # ChromaDB inspector
```
