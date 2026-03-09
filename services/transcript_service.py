"""
services/transcript_service.py
Saves interview transcript to a timestamped text file in logs/.
"""

import os
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class TranscriptService:
    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(config.LOG_DIR, f"interview_{timestamp}.txt")
        self._entries = []
        logger.info(f"Transcript will be saved to: {self.filepath}")

    def add(self, speaker: str, text: str):
        """Add a turn to the transcript. speaker = 'AI' or 'Candidate'."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {speaker}: {text}"
        self._entries.append(entry)
        logger.info(entry)

        if config.SAVE_TRANSCRIPT:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

    def save(self):
        """Force-save all entries (called at end of interview)."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"Interview Transcript — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n".join(self._entries))
        logger.info(f"✓ Transcript saved: {self.filepath}")
