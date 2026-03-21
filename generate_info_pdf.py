"""
Script to generate info.pdf — a comprehensive overview of the Conversational AI Scout project.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame


# ──────────────────────────────────────────────────────────────────────────────
# Custom Colors
# ──────────────────────────────────────────────────────────────────────────────
DARK_BG     = colors.HexColor("#1A1A2E")
ACCENT_BLUE = colors.HexColor("#0F3460")
ACCENT_CYAN = colors.HexColor("#16213E")
HIGHLIGHT   = colors.HexColor("#E94560")
TEXT_DARK   = colors.HexColor("#1A1A2E")
TEXT_LIGHT  = colors.HexColor("#4A4A6A")
TABLE_HEAD  = colors.HexColor("#0F3460")
TABLE_ROW1  = colors.HexColor("#EEF2FF")
TABLE_ROW2  = colors.white
SECTION_BG  = colors.HexColor("#F0F4FF")


# ──────────────────────────────────────────────────────────────────────────────
# Page header/footer callback
# ──────────────────────────────────────────────────────────────────────────────
def header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    w, h = A4

    # Header bar
    canvas_obj.setFillColor(DARK_BG)
    canvas_obj.rect(0, h - 55, w, 55, stroke=0, fill=1)

    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 14)
    canvas_obj.drawString(1.2 * inch, h - 30, "Conversational AI Scout")
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawString(1.2 * inch, h - 46, "RTX 1650 Edition  ·  Technical Project Documentation")

    # Accent stripe
    canvas_obj.setFillColor(HIGHLIGHT)
    canvas_obj.rect(0, h - 58, w, 3, stroke=0, fill=1)

    # Footer
    canvas_obj.setFillColor(TEXT_LIGHT)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(1.2 * inch, 20, "SwitchIt-Pro / conversational-ai-scouts")
    canvas_obj.drawRightString(w - 1.2 * inch, 20, f"Page {doc.page}")
    canvas_obj.setFillColor(HIGHLIGHT)
    canvas_obj.rect(0, 36, w, 1, stroke=0, fill=1)

    canvas_obj.restoreState()


# ──────────────────────────────────────────────────────────────────────────────
# Build PDF
# ──────────────────────────────────────────────────────────────────────────────
def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.2 * inch,
        leftMargin=1.2 * inch,
        topMargin=1.4 * inch,
        bottomMargin=0.9 * inch,
    )

    styles = getSampleStyleSheet()

    # ── Custom Styles ──
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=26,
        textColor=DARK_BG,
        leading=32,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=12,
        textColor=TEXT_LIGHT,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    h1_style = ParagraphStyle(
        "H1",
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.white,
        leading=20,
        spaceBefore=18,
        spaceAfter=4,
        leftIndent=-10,
        rightIndent=-10,
        backColor=DARK_BG,
        borderPad=8,
    )
    h2_style = ParagraphStyle(
        "H2",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=DARK_BG,
        leading=16,
        spaceBefore=14,
        spaceAfter=4,
        borderPad=0,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_DARK,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_DARK,
        leading=14,
        leftIndent=12,
        spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=9,
        textColor=colors.HexColor("#C7254E"),
        backColor=colors.HexColor("#F9F2F4"),
        leading=13,
        leftIndent=12,
        spaceAfter=2,
    )
    note_style = ParagraphStyle(
        "Note",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=TEXT_LIGHT,
        leading=13,
        spaceAfter=4,
    )

    story = []

    # ── Cover / Title ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Conversational AI Scout", title_style))
    story.append(Paragraph("Voice-to-Voice AI Interviewer — RTX 1650 (4 GB VRAM) Edition", subtitle_style))
    story.append(Paragraph("Project: SwitchIt-Pro Screening System &nbsp;|&nbsp; Version 1.0", note_style))
    story.append(HRFlowable(width="100%", thickness=2, color=HIGHLIGHT, spaceAfter=16))

    # ── 1. Project Overview ────────────────────────────────────────────────────
    story.append(Paragraph("  1.  Project Overview", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Conversational AI Scout is a fully local, real-time, voice-to-voice AI interviewer designed to "
        "conduct Round 1 technical screening interviews on behalf of SwitchIt-Pro. The system listens to "
        "a candidate's spoken answers through a microphone, understands what they said, generates an "
        "intelligent reply, and speaks back in a human-sounding voice — all within milliseconds, with "
        "<b>zero cloud API dependencies</b>.",
        body_style
    ))
    story.append(Paragraph(
        "The entire inference pipeline — speech recognition, language reasoning, and voice synthesis — "
        "runs on a single consumer-grade <b>NVIDIA RTX 1650 (4 GB VRAM)</b> GPU, making it accessible "
        "and cost-free to operate while maintaining a natural, real-time conversational pace.",
        body_style
    ))

    # ── 2. Why This Project Exists ─────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  2.  Why This Project Exists", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Running initial HR screening interviews is time-consuming and expensive. This system automates "
        "that first contact, giving every candidate a consistent, friendly, and professionally structured "
        "experience. It:",
        body_style
    ))

    bullets = [
        "Saves recruiter hours by handling repetitive intro-round interviews autonomously.",
        "Ensures every candidate receives the same structured set of questions.",
        "Captures a full timestamped transcript for recruiter review.",
        "Operates entirely offline — no data leaves the machine, ensuring privacy.",
        "Runs cheaply on commodity hardware — no GPU cloud rental required.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• &nbsp;&nbsp;{b}", bullet_style))

    # ── 3. How It Works — The Pipeline ────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  3.  How It Works — The Pipeline", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The system is made up of six tightly choreographed services that pass data from one to the next "
        "in a streaming pipeline. Here is the end-to-end flow:",
        body_style
    ))

    pipeline_data = [
        ["Step", "Service", "Technology", "What It Does"],
        ["1", "Audio Service", "sounddevice", "Captures raw microphone audio in 20 ms chunks and places them into a queue."],
        ["2", "VAD Service", "Silero VAD (CPU)", "Detects whether a chunk contains human speech. Ignores noise; waits for a complete utterance."],
        ["3", "STT Service", "Whisper Small (CUDA)", "Transcribes the finished audio clip into accurate written text using GPU-accelerated inference."],
        ["4", "LLM Service", "Qwen2.5-1.5B Q4 (CUDA)", "Generates the AI interviewer's reply sentence-by-sentence using streaming token generation."],
        ["5", "TTS Service", "Kokoro-82M (CUDA)", "Synthesises each sentence into realistic speech and plays it through the speakers in real time."],
        ["6", "Transcript Service", "Python file I/O", "Silently logs every exchange (speaker + timestamp) to a .txt file in the /logs directory."],
    ]

    pipeline_table = Table(pipeline_data, colWidths=[0.5*inch, 1.2*inch, 1.5*inch, 3.3*inch])
    pipeline_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [TABLE_ROW1, TABLE_ROW2]),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCDD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(pipeline_table)

    # ── 4. Technology & Model Choices ─────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  4.  Technology & Model Choices", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Each model was chosen specifically to fit within a tight 4 GB VRAM envelope while still delivering "
        "production-quality output:",
        body_style
    ))

    vram_data = [
        ["Component", "Model Used", "VRAM", "Why This Model"],
        ["Speech-to-Text (STT)", "Whisper Small (float16)", "~500 MB", "Accurate English ASR at minimal VRAM cost; hallucination filters added."],
        ["Language Model (LLM)", "Qwen2.5-1.5B-Instruct Q4", "~1.2 GB", "4-bit quantised; capable reasoning with small memory footprint."],
        ["Text-to-Speech (TTS)", "Kokoro-82M", "~300 MB", "Ultra-lightweight TTS with natural-sounding voices and <120 ms latency."],
        ["Voice Activity Detection", "Silero VAD (CPU)", "0 MB GPU", "CPU-only; real-time speech/silence classification without GPU overhead."],
        ["OS + Python Overhead", "—", "~2.0 GB", "System and CUDA runtime baseline."],
        ["TOTAL", "", "~4.0 GB ✓", "Fits within RTX 1650 budget with minimal headroom required."],
    ]

    vram_table = Table(vram_data, colWidths=[1.5*inch, 1.6*inch, 0.9*inch, 2.5*inch])
    vram_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2), [TABLE_ROW1, TABLE_ROW2]),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#DDE4FF")),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (-1, -2), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCDD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(vram_table)

    # ── 5. Interview Structure ─────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  5.  Interview Structure & AI Persona", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The AI acts as a friendly, encouraging Technical Interviewer for SwitchIt-Pro. It follows a "
        "fixed 10–15 minute script and asks <b>one question at a time</b>, waiting for the candidate's "
        "full answer before moving on:",
        body_style
    ))

    interview_steps = [
        ("Warm Introduction", "Greets the candidate enthusiastically and asks for a brief self-introduction."),
        ("Resume / Experience", "Asks about the candidate's most recent work or a project they are proud of."),
        ("Basic Technical Question", "Tests a fundamental concept relevant to the role (e.g. Python, web, etc.)."),
        ("Intermediate Technical", "Explores how the candidate would approach a common technical problem."),
        ("Advanced Technical", "Dives into architecture, scalability, or edge-case handling."),
        ("Professional Closing", "Invites questions from the candidate, then wraps up warmly."),
    ]
    for i, (step, desc) in enumerate(interview_steps, 1):
        story.append(Paragraph(f"<b>{i}. {step}</b> — {desc}", bullet_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "All AI responses are capped at 2–3 sentences to keep the conversation brisk and natural. "
        "Affirmations like <i>\"That makes sense!\"</i> or <i>\"Interesting approach!\"</i> are used "
        "before transitioning to the next question.",
        note_style
    ))

    # ── 6. File & Folder Structure ─────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  6.  Project File Structure", h1_style))
    story.append(Spacer(1, 8))

    file_data = [
        ["File / Folder", "Purpose"],
        ["main.py", "WebSocket server entry point. Accepts frontend connections and routes start/stop commands to the orchestrator."],
        ["config.py", "Central settings file. Tweak models, VRAM limits, VAD thresholds, interview persona, and logging here."],
        ["requirements.txt", "All Python dependencies needed to install the project."],
        ["pipeline/orchestrator.py", "Core streaming loop that ties all services together and manages the conversation state machine."],
        ["services/audio_service.py", "Captures microphone audio in real time and feeds it into the pipeline queue."],
        ["services/vad_service.py", "Runs Silero VAD on each audio chunk to detect when the candidate starts and stops speaking."],
        ["services/stt_service.py", "Runs Whisper Small on a completed audio recording to produce a text transcript."],
        ["services/llm_service.py", "Generates the AI interviewer's streaming response using the Qwen2.5-1.5B model."],
        ["services/tts_service.py", "Converts the LLM's streaming text output into playable audio with Kokoro-82M."],
        ["services/transcript_service.py", "Logs each spoken turn to a timestamped text file in the /logs directory."],
        ["logs/", "Auto-created folder. Each interview session saves a transcript as interview_YYYYMMDD_HHMMSS.txt."],
        ["frontend/", "Reserved for a future Web UI frontend."],
    ]

    file_table = Table(file_data, colWidths=[2.1*inch, 4.4*inch])
    file_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [TABLE_ROW1, TABLE_ROW2]),
        ("FONTNAME",      (0, 1), (0, -1), "Courier"),
        ("FONTNAME",      (1, 1), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCDD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(file_table)

    # ── 7. Setup & Running ─────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  7.  Setup & Running", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Prerequisites</b>", h2_style))
    prereqs = [
        "Windows 11 or Ubuntu 22.04+",
        "Python 3.11",
        "NVIDIA CUDA 12.1+ drivers (RTX 1650 with driver 545+)",
    ]
    for p in prereqs:
        story.append(Paragraph(f"• &nbsp;&nbsp;{p}", bullet_style))

    story.append(Paragraph("<b>Installation</b>", h2_style))
    install_steps = [
        ("Install PyTorch with CUDA 12.1:",
         "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"),
        ("Install all project dependencies:",
         "pip install -r requirements.txt"),
        ("Start the server (models auto-download ~2 GB on first run):",
         "python main.py"),
    ]
    for desc, cmd in install_steps:
        story.append(Paragraph(desc, bullet_style))
        story.append(Paragraph(cmd, code_style))
        story.append(Spacer(1, 3))

    story.append(Paragraph("<b>Usage</b>", h2_style))
    story.append(Paragraph(
        "Once the server is running at <b>ws://localhost:8765</b>, a frontend client can connect and send "
        "a <i>{\"command\": \"start\"}</i> message. Speak naturally into the microphone; the AI will "
        "respond through the speakers. Send <i>{\"command\": \"stop\"}</i> or press <b>Ctrl+C</b> to end "
        "the session. A full transcript is automatically saved to the <b>logs/</b> directory.",
        body_style
    ))

    # ── 8. Key Tuning Parameters ───────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  8.  Key Configuration Parameters", h1_style))
    story.append(Spacer(1, 8))

    param_data = [
        ["Parameter", "File", "Default", "Effect"],
        ["VAD_THRESHOLD", "config.py", "0.85", "Minimum speech probability (0–1). Higher = stricter, fewer false triggers."],
        ["VAD_SILENCE_MS", "config.py", "1500 ms", "How long of silence to wait before treating an utterance as complete."],
        ["LLM_MAX_NEW_TOKENS", "config.py", "200", "Max tokens in the AI's reply. Reduce to save VRAM; increase for longer answers."],
        ["LLM_TEMPERATURE", "config.py", "0.7", "Creativity of replies. Lower = more deterministic; higher = more varied."],
        ["TTS_VOICE", "config.py", "af_sarah", "Voice character. Options: af_sarah, am_adam, bf_emma."],
        ["TTS_SPEED", "config.py", "1.0", "Speech rate multiplier. 1.0 is normal speed."],
        ["SYSTEM_PROMPT", "config.py", "(long)", "Full persona and interview structure instructions for the AI."],
        ["SAVE_TRANSCRIPT", "config.py", "True", "Enable/disable saving conversation logs to the /logs folder."],
    ]

    param_table = Table(param_data, colWidths=[1.5*inch, 0.9*inch, 0.9*inch, 3.2*inch])
    param_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [TABLE_ROW1, TABLE_ROW2]),
        ("FONTNAME",      (0, 1), (0, -1), "Courier"),
        ("FONTNAME",      (1, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCDD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(param_table)

    # ── 9. Design Decisions / Comparisons ─────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  9.  Design Decisions vs. High-End Alternatives", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The original specification targeted RTX 3090/4090 (24 GB) hardware. This project adapts those "
        "choices for a 4 GB VRAM constraint:",
        body_style
    ))

    compare_data = [
        ["Original Spec (24GB GPU)", "This Project (4GB RTX 1650)", "Reason for Change"],
        ["Canary Qwen 2.5B STT", "Whisper Small", "2.5B STT alone needs 3GB+; Whisper Small is just as accurate."],
        ["Qwen3-7B Instruct LLM", "Qwen2.5-1.5B Q4", "7B model needs 8GB+ even quantised; 1.5B Q4 fits in 1.2GB."],
        ["CosyVoice 0.5B TTS", "Kokoro-82M TTS", "Kokoro is lighter, easier to install on Windows, and equally natural."],
        ["LiveKit + Pipecat WebRTC", "sounddevice + websockets", "Full WebRTC stack is overkill for a single-machine local session."],
    ]

    compare_table = Table(compare_data, colWidths=[1.8*inch, 1.8*inch, 3.0*inch])
    compare_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [TABLE_ROW1, TABLE_ROW2]),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCDD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(compare_table)

    # ── 10. Summary ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("  10.  Summary", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Conversational AI Scout is a production-ready, fully offline voice-to-voice AI interviewer. "
        "It autonomously conducts structured, friendly, and professional Round 1 technical screening "
        "interviews by listening, understanding, and speaking back in real time. Built to run entirely "
        "on a consumer RTX 1650 GPU, it eliminates cloud costs and privacy concerns while delivering a "
        "seamless candidate experience. Every interview session is automatically transcribed and saved "
        "for recruiter review.",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=HIGHLIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated automatically · SwitchIt-Pro / conversational-ai-scouts · 2026",
        note_style
    ))

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"✅  PDF created: {output_path}")


if __name__ == "__main__":
    build_pdf("info.pdf")
