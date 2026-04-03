"""
services/ollama_client.py
-------------------------
A dedicated, lightweight client to connect to local Ollama.
Replaces the old dependency on the scout_ai_interviewer Qwen script.
"""

import json
import logging
import requests
from typing import Tuple

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, model_name: str = None):
        self.model = model_name or getattr(config, "LLM_MODEL", "qwen2.5:7b")
        self.base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"
        
        logger.info(f"OllamaClient initialized for model: {self.model} at {self.base_url}")
        
    def prompt(self, prompt_text: str, temperature: float = None) -> Tuple[str, None]:
        """
        Send a generic prompt to Ollama and return the text response.
        Returns (response_text, None) to maintain compatibility with legacy code calling `raw, _ = `.
        """
        temp = temperature if temperature is not None else getattr(config, "LLM_TEMPERATURE", 0.7)
        
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temp,
                "num_predict": getattr(config, "LLM_MAX_NEW_TOKENS", 200)
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip(), None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection failed: {e}")
            raise RuntimeError(f"Failed to communicate with Ollama: {e}")

