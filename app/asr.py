"""Whisper transcription for the live captioning app (PLAN.md D5: HF transformers).

The model defaults to seanghay/whisper-small-khmer-v2. To use our fine-tuned model:
    ASR_MODEL=results/run1/model python app/server.py
"""

import os

import numpy as np
import soxr
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

ASR_MODEL = os.environ.get("ASR_MODEL", "seanghay/whisper-small-khmer-v2")
BASE_MODEL = "openai/whisper-small"
SR = 16000
SILENCE_RMS = 0.01  # skip near-silent chunks; Whisper invents text on silence
MAX_NEW_TOKENS = 250  # Khmer is ~2 Whisper tokens per character; 120 cut off a 7 s clip


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Transcriber:
    def __init__(self, model_name=ASR_MODEL):
        self.name = model_name
        self.device = pick_device()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device).eval()
        # seanghay's generation config predates the language= argument; reuse the base model's
        self.model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)

    @torch.no_grad()
    def __call__(self, audio, sr=SR):
        audio = np.asarray(audio, dtype=np.float32)
        if sr != SR:
            audio = soxr.resample(audio, sr, SR)
        if audio.size == 0 or np.sqrt(np.mean(audio ** 2)) < SILENCE_RMS:
            return ""
        feats = self.processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
        out = self.model.generate(feats.to(self.device), language="khmer", task="transcribe",
                                  max_new_tokens=MAX_NEW_TOKENS)
        return self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
