import os
import uuid
import json
import logging
import asyncio
import threading
from datetime import datetime
from typing import Optional

import httpx
import ollama
import chromadb
import pdfplumber
import openpyxl

from fastapi import FastAPI, UploadFile, Form, File, BackgroundTasks, Request
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration — 8-column schema
EXCEL_FILE = "data/questions.xlsx"
HEADERS = [
    "question_id", "question_text", "role",
    "evaluation_area", "experience_level",
    "question_type", "what_ai_listens_for", "follow_up_trigger"
]
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"

# Global state
job_events = {}  # { job_id: [event_dict, ...] }

# ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_store")
scout_questions = chroma_client.get_or_create_collection(name="scout_questions")
jd_registry = chroma_client.get_or_create_collection(name="jd_registry")

@app.on_event("startup")
async def startup_event():
    # Ping Ollama
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(OLLAMA_HOST)
            if resp.status_code != 200:
                logger.warning(f"Ollama reachable but returned {resp.status_code}")
            else:
                logger.info("Ollama is reachable.")
    except Exception as e:
        logger.warning(f"Ollama is unreachable at {OLLAMA_HOST}: {e}")

    # Create Excel if missing
    if not os.path.exists(EXCEL_FILE):
        os.makedirs("data", exist_ok=True)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Questions"
        ws.append(HEADERS)
        wb.save(EXCEL_FILE)
        logger.info(f"Created {EXCEL_FILE}")

def emit_event(job_id: str, event: dict):
    if job_id not in job_events:
        job_events[job_id] = []
    job_events[job_id].append(event)
    logger.info(f"Job {job_id} event: {event}")

# --- Helper logic ---

def extract_text_from_file(file_path: str, filename: str) -> str:
    text = ""
    if filename.lower().endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text.strip()

def get_next_ids(num_questions: int):
    """Auto-increment question IDs (Q0001, Q0002…) and variant group ID."""
    wb = openpyxl.load_workbook(EXCEL_FILE)
    ws = wb.active
    
    max_q_num = 0
    max_vg_num = 0
    
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row: continue
        q_id = row[0]  # col A = question_id
        # variant group stored in jd_registry, but we still track Q-numbers
        if q_id and str(q_id).startswith("Q"):
            try:
                num = int(str(q_id)[1:])
                max_q_num = max(max_q_num, num)
            except: pass

    # VG counter comes from jd_registry count
    try:
        max_vg_num = jd_registry.count()
    except: pass

    next_vg_id = f"VG{(max_vg_num + 1):04d}"
    next_q_ids = [f"Q{(max_q_num + i + 1):04d}" for i in range(num_questions)]
    return next_vg_id, next_q_ids

def call_ollama_json(prompt: str, system_msg: str, retries=1):
    for attempt in range(retries + 1):
        try:
            resp = ollama.generate(
                model=LLM_MODEL,
                system=system_msg,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0.5,
                    "num_ctx": 8192,
                    "num_predict": 3000
                }
            )
            return json.loads(resp['response'])
        except Exception as e:
            logger.error(f"Ollama call failed (attempt {attempt+1}): {e}")
            system_msg += "\nReturn ONLY the raw JSON object, no markdown, no backticks, no explanation."
    raise Exception("Failed to parse JSON from LLM after retries.")

def get_embedding(text: str):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp['embedding']

# --- Background Worker ---

def process_job(
    job_id: str,
    jd_raw_text: str,
    file_path: Optional[str],
    original_filename: Optional[str],
    role: str,
    experience_level: str,
    num_questions: int
):
    try:
        # STEP 1
        emit_event(job_id, {"step": "Preparing job context...", "progress": 5})
        jd_text = jd_raw_text
        if file_path and original_filename:
            jd_text = extract_text_from_file(file_path, original_filename)
        
        jd_text = jd_text.strip() if jd_text else ""
        
        # If no JD provided, synthesize one from role + level so the pipeline still works
        if not jd_text or len(jd_text) < 50:
            emit_event(job_id, {"step": "No JD provided — generating generic context via Qwen...", "progress": 8})
            synth_prompt = f"Write a concise job description (150-200 words) for a {experience_level} {role} position at a modern tech company."
            synth_sys = "You are an HR manager. Write only the job description text, no extra commentary."
            try:
                resp = ollama.generate(model=LLM_MODEL, system=synth_sys, prompt=synth_prompt, options={"num_predict": 400})
                jd_text = resp['response'].strip()
            except Exception as e:
                jd_text = f"We are looking for a {experience_level} {role} to join our growing team. The candidate should have strong communication, problem-solving, and domain expertise relevant to {role} responsibilities."
            
        # STEP 2
        emit_event(job_id, {"step": "Checking for duplicate jobs...", "progress": 10})
        
        jd_embedding = get_embedding(jd_text)
        
        # Check jd_registry
        possibly_similar = False
        matched_variant_group_id = None
        matched_role = None
        similarity_score = None
        
        if jd_registry.count() > 0:
            results = jd_registry.query(
                query_embeddings=[jd_embedding],
                n_results=5,
                include=['distances', 'metadatas']
            )
            if results and results['distances'] and len(results['distances'][0]) > 0:
                best_dist = results['distances'][0][0]
                best_meta = results['metadatas'][0][0]
                
                if best_dist < 0.15:
                    emit_event(job_id, {
                        "step": "Duplicate detected",
                        "progress": 100,
                        "duplicate": True,
                        "message": "A similar job already exists in the system.",
                        "matched_variant_group_id": best_meta.get('variant_group_id'),
                        "matched_role": best_meta.get('role'),
                        "similarity_score": round(1 - best_dist, 2)
                    })
                    return # STOP processing
                elif 0.15 <= best_dist < 0.30:
                    possibly_similar = True
                    matched_variant_group_id = best_meta.get('variant_group_id')
                    matched_role = best_meta.get('role')
                    similarity_score = round(1 - best_dist, 2)

        # STEP 3
        emit_event(job_id, {"step": "Extracting skills from context...", "progress": 25})
        
        sys_msg_3 = """You are an expert HR analyst. Given a job description, extract:
1. A list of 5-10 core technical and soft skills required
2. The primary evaluation areas (pick only from: Communication Skills, Conceptual Clarity, Problem Solving, Role Relevance, Resume Authenticity)
3. Key responsibilities in 3 bullet points
Return ONLY valid JSON with no markdown, no backticks:
{ "skills": [], "evaluation_areas": [], "responsibilities": [] }"""
        
        extraction = call_ollama_json(jd_text, sys_msg_3)
        skills = extraction.get("skills", [])
        eval_areas = extraction.get("evaluation_areas", [])
        responsibilities = extraction.get("responsibilities", [])
        
        # STEP 4 — Batch generation (10 per batch to fit qwen2.5:1.5b context)
        BATCH_SIZE = 10
        num_batches = (num_questions + BATCH_SIZE - 1) // BATCH_SIZE
        generated_qs = []

        for batch_num in range(num_batches):
            batch_target = min(BATCH_SIZE, num_questions - len(generated_qs))
            already_done = len(generated_qs)
            emit_event(job_id, {
                "step": f"Generating batch {batch_num+1}/{num_batches} ({batch_target} questions)...",
                "progress": 45 + int(20 * batch_num / num_batches)
            })

            # Tell Qwen which evaluation areas to cover in this batch
            all_areas = ["Communication Skills", "Conceptual Clarity", "Problem Solving", "Role Relevance", "Resume Authenticity"]
            batch_areas = all_areas[already_done % 5 : already_done % 5 + batch_target] or all_areas[:batch_target]

            sys_msg_4 = f"""You are an expert interview question designer for {role} positions.
Generate exactly {batch_target} interview questions for a {experience_level} candidate.
Context — skills required: {skills}. Key responsibilities: {responsibilities}.
Preferred evaluation areas for this batch: {batch_areas}.

Return a JSON array of exactly {batch_target} objects. Each object must have EXACTLY these 8 fields:
[
  {{
    "question_id": "",
    "question_text": "specific realistic question for {experience_level} {role}",
    "role": "{role}",
    "evaluation_area": "one of: Communication Skills | Conceptual Clarity | Problem Solving | Role Relevance | Resume Authenticity",
    "experience_level": "{experience_level}",
    "question_type": "one of: Behavioral | Situational | Profile-Grounded | Follow-up | Knowledge Check",
    "what_ai_listens_for": "1-2 concrete sentences on signals to score positively",
    "follow_up_trigger": "phrase under 10 words triggering a follow-up probe"
  }}
]
Return ONLY the raw JSON array. No markdown, no backticks, no explanation."""

            try:
                batch_qs = call_ollama_json("Generate the questions now.", sys_msg_4)
                if isinstance(batch_qs, dict) and "questions" in batch_qs:
                    batch_qs = batch_qs["questions"]
                if isinstance(batch_qs, list):
                    generated_qs.extend(batch_qs[:batch_target])
            except Exception as e:
                logger.warning(f"Batch {batch_num+1} failed: {e}")

            if len(generated_qs) >= num_questions:
                break

        generated_qs = generated_qs[:num_questions]
        
        # STEP 5 — Save to Excel (8 columns)
        emit_event(job_id, {"step": "Saving to Excel (8-column schema)...", "progress": 65})
        next_vg_id, next_q_ids = get_next_ids(len(generated_qs))

        wb = openpyxl.load_workbook(EXCEL_FILE)
        ws = wb.active

        chroma_ids = []
        chroma_embeddings = []
        chroma_documents = []
        chroma_metadatas = []

        for idx, q in enumerate(generated_qs):
            q_id = next_q_ids[idx]
            q_text = q.get("question_text", "").strip()
            eval_area = q.get("evaluation_area", "Role Relevance")
            q_type    = q.get("question_type", "Behavioral")
            listens   = q.get("what_ai_listens_for", "")
            trigger   = q.get("follow_up_trigger", "")

            # Write exactly 8 columns
            row = [
                q_id,          # question_id
                q_text,        # question_text
                role,          # role
                eval_area,     # evaluation_area
                experience_level,  # experience_level
                q_type,        # question_type
                listens,       # what_ai_listens_for
                trigger,       # follow_up_trigger
            ]
            ws.append(row)

            # Embedding text = question + what AI listens for (most semantic signal)
            embedding_text = f"{q_text}. {listens}".strip()

            chroma_ids.append(q_id)
            chroma_documents.append(embedding_text)
            chroma_metadatas.append({
                "role": role,
                "evaluation_area": eval_area,
                "experience_level": experience_level,
                "question_type": q_type,
                "follow_up_trigger": trigger,
                "variant_group_id": next_vg_id,
            })

        # Save with retry — file may be open in Excel on Windows
        import time
        for attempt in range(10):
            try:
                wb.save(EXCEL_FILE)
                break
            except PermissionError:
                if attempt == 9:
                    raise PermissionError(
                        f"Cannot save '{EXCEL_FILE}' — please CLOSE the file in Excel and retry."
                    )
                emit_event(job_id, {
                    "step": f"⚠️ Excel file is open — please close it. Retrying in 3s... ({attempt+1}/10)",
                    "progress": 65
                })
                time.sleep(3)


        # STEP 6 — Embed and push to ChromaDB
        emit_event(job_id, {"step": f"Embedding {len(chroma_ids)} questions into ChromaDB...", "progress": 80})
        for i, doc in enumerate(chroma_documents):
            chroma_embeddings.append(get_embedding(doc))

        scout_questions.upsert(
            ids=chroma_ids,
            embeddings=chroma_embeddings,
            documents=chroma_documents,
            metadatas=chroma_metadatas
        )

        # STEP 7
        emit_event(job_id, {"step": "Registering JD...", "progress": 92})
        jd_registry.upsert(
            ids=[next_vg_id],
            embeddings=[jd_embedding],
            documents=[jd_text[:500]],
            metadatas=[{
                "role": role,
                "experience_level": experience_level,
                "variant_group_id": next_vg_id,
                "created_at": datetime.utcnow().isoformat(),
                "skills_extracted": json.dumps(skills)
            }]
        )

        # STEP 8
        emit_event(job_id, {
            "step": "Done",
            "progress": 100,
            "questions_added": len(generated_qs),
            "skills_extracted": skills,
            "variant_group_id": next_vg_id,
            "possibly_similar": possibly_similar,
            "matched_variant_group_id": matched_variant_group_id,
            "matched_role": matched_role,
            "similarity_score": similarity_score
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        emit_event(job_id, {"step": "Error", "progress": 0, "error": str(e)})

    finally:
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass

# --- HTTP Routes ---

@app.get("/")
def index():
    return FileResponse("portal.html")

@app.get("/api/health")
def health():
    # check ollama
    ollama_ok = False
    try:
        r = httpx.get(OLLAMA_HOST, timeout=2.0)
        ollama_ok = (r.status_code == 200)
    except: pass
    return {"ollama": ollama_ok, "chroma": True}

@app.post("/api/generate")
async def generate_questions(
    role: str = Form(...),
    experience_level: str = Form(...),
    num_questions: int = Form(20),
    jd_text: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    job_id = str(uuid.uuid4())
    job_events[job_id] = []
    
    file_path = None
    original_filename = None
    if file and file.filename:
        file_path = f"/tmp/{job_id}_{file.filename}"
        os.makedirs("/tmp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        original_filename = file.filename

    thread = threading.Thread(
        target=process_job,
        args=(job_id, jd_text, file_path, original_filename, role, experience_level, num_questions)
    )
    thread.start()
    
    return {"job_id": job_id}

@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str, request: Request):
    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break
            if job_id in job_events:
                events = job_events[job_id]
                while last_idx < len(events):
                    ev = events[last_idx]
                    yield {"data": json.dumps(ev)}
                    if ev.get("progress") == 100 or ev.get("step") == "Error":
                        return
                    last_idx += 1
            await asyncio.sleep(0.5)
            
    return EventSourceResponse(event_generator())

@app.get("/api/jobs")
def list_jobs():
    if jd_registry.count() == 0:
        return []

    results = jd_registry.get()
    ids       = results.get("ids", [])
    metadatas = results.get("metadatas", [])

    # Count questions per VG from ChromaDB (safe iteration — old docs may have no metadata)
    counts = {}
    try:
        all_q = scout_questions.get(include=["metadatas"])
        for meta in (all_q.get("metadatas") or []):
            if not isinstance(meta, dict):
                continue
            vg = meta.get("variant_group_id")
            if vg:
                counts[vg] = counts.get(vg, 0) + 1
    except Exception as e:
        logger.warning(f"Could not count questions from ChromaDB: {e}")

    jobs = []
    for vg_id, m in zip(ids, metadatas):
        if not m: continue
        vg = m.get("variant_group_id", vg_id)
        try:
            formatted_date = datetime.fromisoformat(m.get("created_at", "")).strftime("%d %b %Y")
        except:
            formatted_date = m.get("created_at", "")

        skills_str = m.get("skills_extracted", "[]")
        try:
            skills = json.loads(skills_str)
        except:
            skills = []

        jobs.append({
            "variant_group_id": vg,
            "role": m.get("role", "Unknown"),
            "experience_level": m.get("experience_level", ""),
            "created_at": formatted_date,
            "num_questions": counts.get(vg, 0),
            "skills_extracted": skills
        })

    jobs.sort(key=lambda x: x.get("variant_group_id", ""), reverse=True)
    return jobs

@app.delete("/api/jobs/{variant_group_id}")
def delete_job(variant_group_id: str):
    q_removed = 0

    # Get role+level from jd_registry to identify Excel rows (new 8-col schema has no vg_id in Excel)
    target_role = None
    target_level = None
    try:
        reg = jd_registry.get(ids=[variant_group_id], include=["metadatas"])
        if reg["metadatas"]:
            target_role  = reg["metadatas"][0].get("role")
            target_level = reg["metadatas"][0].get("experience_level")
    except: pass

    # Delete matching rows from Excel by role + experience_level (cols 2 and 4, 0-indexed)
    if os.path.exists(EXCEL_FILE) and target_role and target_level:
        wb = openpyxl.load_workbook(EXCEL_FILE)
        ws = wb.active
        for i in range(ws.max_row, 1, -1):
            row_role  = ws.cell(row=i, column=3).value  # col C = role
            row_level = ws.cell(row=i, column=5).value  # col E = experience_level
            if row_role == target_role and row_level == target_level:
                ws.delete_rows(i)
                q_removed += 1
        wb.save(EXCEL_FILE)

    try:
        scout_questions.delete(where={"variant_group_id": variant_group_id})
    except: pass

    try:
        jd_registry.delete(ids=[variant_group_id])
    except: pass

    return {"deleted": True, "questions_removed": q_removed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8050, log_level="info")
