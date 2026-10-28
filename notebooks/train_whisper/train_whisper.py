"""Kaggle GPU kernel: Whisper run 1 (PLAN.md).

Run 1 settings (decided Oct 26):
  - start from vanilla openai/whisper-small
  - train on FLEURS km_kh train + SLR42 (SLR42 word spaces removed)
  - labels keep punctuation and digits as written (other options compared after Oct 28)
  - select the best checkpoint by CER on FLEURS dev
  - report CER on FLEURS test for: vanilla whisper-small, seanghay/whisper-small-khmer-v2, our model

Push with ./push.sh (bundles src/normalize.py into this file). ./push.sh --smoke does a short test run.

Output (/kaggle/working/):
  model/                    best fine-tuned model + processor
  results/cer_summary.csv   model, test set, CER overall and by gender
  results/pred_*.csv        per-clip reference, hypothesis, CER
  results/log_history.json  training loss and dev CER per eval step
"""

import csv
import glob
import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile

subprocess.run(["pip", "install", "-q", "jiwer"], check=True)

import jiwer
import numpy as np
import soundfile as sf
import soxr
import torch
from huggingface_hub import hf_hub_download
from transformers import (GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from normalize import clean_text, normalize_for_cer

SMOKE_TEST = False  # push.sh --smoke sets this to True

BASE_MODEL = "openai/whisper-small"
BASELINES = ["openai/whisper-small", "seanghay/whisper-small-khmer-v2"]
LANGUAGE, TASK = "khmer", "transcribe"
SR = 16000
MAX_AUDIO_S = 30.0
MAX_LABEL_TOKENS = 448

MAX_STEPS = 1000  # Oct 28: cut from 3000 to finish before the progress presentation; 3000-step run after
EVAL_STEPS = 500
LEARNING_RATE = 1e-5
WARMUP_STEPS = 300
EFFECTIVE_BATCH = 16

DATA = "/kaggle/tmp/data"
CKPT = "/kaggle/tmp/ckpt"
OUT = "/kaggle/working"
SLR42_URLS = ["https://www.openslr.org/resources/42/km_kh_male.zip",
              "https://openslr.trmal.net/resources/42/km_kh_male.zip"]

if SMOKE_TEST:
    MAX_STEPS, EVAL_STEPS, WARMUP_STEPS = 40, 20, 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------- data

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    return soxr.resample(audio, sr, SR) if sr != SR else audio


def fleurs_split(split):
    tsv = hf_hub_download("google/fleurs", f"data/km_kh/{split}.tsv", repo_type="dataset")
    tar_path = hf_hub_download("google/fleurs", f"data/km_kh/audio/{split}.tar.gz", repo_type="dataset")
    with tarfile.open(tar_path) as tar:
        tar.extractall(f"{DATA}/fleurs")
    rows = []
    with open(tsv, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"path": f"{DATA}/fleurs/{split}/{r[1]}", "text": clean_text(r[2]),
                         "raw": r[2], "gender": r[6].lower(), "source": "fleurs"})
    return rows


def slr42():
    zip_path = f"{DATA}/km_kh_male.zip"
    for url in SLR42_URLS:
        if subprocess.run(["wget", "-q", "-O", zip_path, url]).returncode == 0:
            break
    else:
        raise RuntimeError("SLR42 download failed from all mirrors")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
    rows = []
    with open(f"{DATA}/km_kh_male/line_index.tsv", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.rstrip("\n").split("\t")
            rows.append({"path": f"{DATA}/km_kh_male/wavs/{utt_id}.wav", "text": clean_text(text, remove_spaces=True),
                         "raw": text, "gender": "male", "source": "slr42"})
    return rows


def attach_audio(rows, tokenizer, name):
    kept, dropped = [], {"too_long_audio": 0, "too_long_label": 0, "empty": 0}
    for r in rows:
        if not r["text"]:
            dropped["empty"] += 1
            continue
        audio = load_audio(r["path"])
        if len(audio) / SR > MAX_AUDIO_S:
            dropped["too_long_audio"] += 1
            continue
        labels = tokenizer(r["text"]).input_ids
        if len(labels) > MAX_LABEL_TOKENS:
            dropped["too_long_label"] += 1
            continue
        kept.append({**r, "audio": audio, "labels": labels})
    hours = sum(len(r["audio"]) for r in kept) / SR / 3600
    print(f"{name}: kept {len(kept)} clips ({hours:.2f} h), dropped {dropped}")
    return kept


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, rows, feature_extractor):
        self.rows, self.fe = rows, feature_extractor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = self.fe(r["audio"], sampling_rate=SR).input_features[0]
        return {"input_features": feats, "labels": r["labels"]}


class Collator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor, self.start = processor, decoder_start_token_id

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.start).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


# ---------------------------------------------------------------- evaluation

def cer(refs, hyps):
    pairs = [(normalize_for_cer(r), normalize_for_cer(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    return jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])


@torch.no_grad()
def transcribe(model, processor, rows, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    hyps = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        feats = processor.feature_extractor([r["audio"] for r in chunk], sampling_rate=SR,
                                            return_tensors="pt").input_features
        out = model.generate(feats.to(device, dtype=model.dtype), language=LANGUAGE, task=TASK, max_new_tokens=440)
        hyps.extend(processor.tokenizer.batch_decode(out, skip_special_tokens=True))
    return hyps


def evaluate_and_save(model, processor, rows, model_name, set_name, summary):
    start = time.time()
    hyps = transcribe(model, processor, rows)
    refs = [r["text"] for r in rows]
    safe = model_name.replace("/", "__")
    with open(f"{OUT}/results/pred_{safe}_{set_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "gender", "reference", "hypothesis", "cer"])
        for r, h in zip(rows, hyps):
            w.writerow([os.path.basename(r["path"]), r["gender"], r["text"], h,
                        f"{cer([r['text']], [h]):.4f}" if normalize_for_cer(r["text"]) else ""])
    row = {"model": model_name, "test_set": set_name, "n": len(rows), "cer": cer(refs, hyps)}
    for g in ("male", "female"):
        idx = [i for i, r in enumerate(rows) if r["gender"] == g]
        row[f"cer_{g}"] = cer([refs[i] for i in idx], [hyps[i] for i in idx]) if idx else ""
    summary.append(row)
    print(f"{model_name} on {set_name}: CER {row['cer']:.4f} ({(time.time() - start) / 60:.1f} min)")
    for r, h in list(zip(refs, hyps))[:3]:
        print("  REF:", r, "\n  HYP:", h)


def write_summary(summary):
    with open(f"{OUT}/results/cer_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_set", "n", "cer", "cer_male", "cer_female"])
        w.writeheader()
        w.writerows(summary)


# ---------------------------------------------------------------- main

def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(f"{OUT}/results", exist_ok=True)
    n_gpu = torch.cuda.device_count()
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n_gpu)], "| smoke test:", SMOKE_TEST)

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    tok = processor.tokenizer

    train_rows = fleurs_split("train") + slr42()
    dev_rows, test_rows = fleurs_split("dev"), fleurs_split("test")
    if SMOKE_TEST:
        train_rows, dev_rows, test_rows = train_rows[::40], dev_rows[:32], test_rows[:32]
    train = attach_audio(train_rows, tok, "train (FLEURS train + SLR42)")
    dev = attach_audio(dev_rows, tok, "dev (FLEURS dev)")
    test = attach_audio(test_rows, tok, "test (FLEURS test)")

    summary = []

    # Baselines first, so their numbers exist even if training fails.
    for name in BASELINES:
        m = WhisperForConditionalGeneration.from_pretrained(name).to(DEVICE, dtype=DTYPE)
        # seanghay's generation config predates the language= argument; reuse the base model's
        m.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
        evaluate_and_save(m, WhisperProcessor.from_pretrained(name), test, name, "fleurs_test", summary)
        del m
        torch.cuda.empty_cache()
    write_summary(summary)

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
        hyps = tok.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tok.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer(refs, hyps)}

    per_device = max(1, EFFECTIVE_BATCH // max(1, n_gpu))
    args = Seq2SeqTrainingArguments(
        output_dir=CKPT,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=DEVICE == "cuda",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_TOKENS,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=SpeechDataset(train, processor.feature_extractor),
        eval_dataset=SpeechDataset(dev, processor.feature_extractor),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    start = time.time()
    trainer.train()
    print(f"training took {(time.time() - start) / 60:.1f} min")

    with open(f"{OUT}/results/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    trainer.save_model(f"{OUT}/model")
    processor.save_pretrained(f"{OUT}/model")

    best = trainer.model.to(DEVICE, dtype=DTYPE).eval()
    best.config.use_cache = True
    evaluate_and_save(best, processor, test, "run1_whisper-small_fleurs+slr42", "fleurs_test", summary)
    write_summary(summary)
    print(open(f"{OUT}/results/cer_summary.csv").read())
    shutil.rmtree(CKPT, ignore_errors=True)


if __name__ == "__main__":
    main()
