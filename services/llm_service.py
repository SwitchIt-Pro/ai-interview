"""
services/llm_service.py
LLM using Qwen2.5-1.5B-Instruct with 4-bit quantization
Optimized for RTX 1650 (4GB VRAM) — ~1.2GB VRAM usage
"""

import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread
import config

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        """
        Sets up the Large Language Model (LLM) service.
        Think of this as awakening the "brain" of our AI interviewer.
        It loads the necessary files and sets up a blank slate for the conversation history.
        """
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self._load_model()
        self._init_conversation()

    def _load_model(self):
        """
        Loads the core AI model into the computer's graphics card (GPU).
        We use special compression techniques (4-bit quantization) to squeeze a large, smart model 
        into a smaller amount of memory so it runs smoothly on everyday home computers.
        """
        logger.info(f"Loading LLM: {config.LLM_MODEL_ID} (4-bit quant)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,     # saves ~0.2GB extra
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_ID,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("✓ LLM model loaded.")

    def _init_conversation(self):
        """
        Starts a fresh conversation memory.
        It gives the AI its initial instructions (the "system prompt"), like reminding it
        that it's acting as an interviewer and what rules it needs to follow.
        """
        self.conversation_history = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]

    def generate_stream(self, user_text: str):
        """
        Takes what the user just said and asks the AI to think of a reply.
        
        Instead of waiting for the AI to type out the entire paragraph before speaking,
        this function streams the AI's answer word-by-word (or small chunk by chunk).
        This way, the voice system can start reading the first sentence out loud while the AI 
        is still thinking about the rest, making it feel like a real-time natural conversation.
        """
        self.conversation_history.append({"role": "user", "content": user_text})

        # Build prompt using Qwen chat template
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            do_sample=True,
            streamer=streamer,
        )

        # Run generation in background thread so we can yield from streamer
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        buffer = ""

        for chunk in streamer:
            buffer += chunk
            full_response += chunk

            # Yield to TTS every ~5 words (natural sentence fragment)
            if len(buffer.split()) >= 5 or any(p in buffer for p in ".!?,"):
                yield buffer
                buffer = ""

        # Flush any remaining buffer
        if buffer.strip():
            yield buffer

        thread.join()

        # Save assistant turn for multi-turn memory
        self.conversation_history.append(
            {"role": "assistant", "content": full_response.strip()}
        )
        logger.debug(f"LLM response: '{full_response.strip()[:80]}...'")

    def reset(self):
        """
        Wipes the AI's short-term memory of what was just talked about.
        Used when starting a brand new interview so the AI doesn't remember the previous candidate.
        """
        self._init_conversation()
        logger.info("Conversation history reset.")
