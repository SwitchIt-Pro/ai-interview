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
        """
        Sets up the Transcript Service.
        Think of this as grabbing a fresh notepad and a pen right before the interview starts.
        It creates a new text file named with today's date and time so we can save the conversation.
        """
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(config.LOG_DIR, f"interview_{timestamp}.txt")
        self._entries = []
        logger.info(f"Transcript will be saved to: {self.filepath}")

    def add(self, speaker: str, text: str):
        """
        Takes a single sentence or paragraph from either the AI or the Candidate and writes it down.
        It adds a timestamp (e.g., [14:05:01]) so we know exactly when it was said, and then safely
        appends it to our text file.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {speaker}: {text}"
        self._entries.append(entry)
        logger.info(entry)

        if config.SAVE_TRANSCRIPT:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

    def save(self):
        """
        Finalizes and saves the entire interview log document.
        Think of this as adding a nice header to our notepad and storing it securely 
        in a file cabinet when the interview is completely finished.
        """
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"Interview Transcript — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n".join(self._entries))
        logger.info(f"✓ Transcript saved: {self.filepath}")
