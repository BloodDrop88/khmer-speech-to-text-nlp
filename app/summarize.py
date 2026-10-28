"""Rolling Khmer summarization with songhieng/khmer-mt5-summarization.

The model reads at most 512 tokens. Longer transcripts are packed into pieces that fit,
each piece is summarized, and the joined partial summaries are summarized again.
"""

import os
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from asr import pick_device

SUM_MODEL = os.environ.get("SUM_MODEL", "songhieng/khmer-mt5-summarization")
MAX_INPUT_TOKENS = 512
PIECE_TOKENS = 480  # leaves room for the "summarize: " prefix
MAX_LEVELS = 3
SEGMENT = re.compile(r"(?<=[។៕])|\s+")


class Summarizer:
    def __init__(self, model_name=SUM_MODEL):
        self.name = model_name
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def n_tokens(self, text):
        return len(self.tokenizer(text).input_ids)

    def pieces(self, text):
        out, current = [], ""
        for seg in filter(None, (s.strip() for s in SEGMENT.split(text))):
            candidate = f"{current} {seg}".strip()
            if current and self.n_tokens(candidate) > PIECE_TOKENS:
                out.append(current)
                current = seg
            else:
                current = candidate
        if current:
            out.append(current)
        return out

    @torch.no_grad()
    def summarize_piece(self, text):
        inputs = self.tokenizer(f"summarize: {text}", return_tensors="pt", truncation=True,
                                max_length=MAX_INPUT_TOKENS).to(self.device)
        ids = self.model.generate(**inputs, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    def __call__(self, transcript):
        text = transcript.strip()
        for _ in range(MAX_LEVELS):
            parts = self.pieces(text)
            if len(parts) <= 1:
                return self.summarize_piece(text) if text else ""
            text = " ".join(self.summarize_piece(p) for p in parts)
        return self.summarize_piece(text)
