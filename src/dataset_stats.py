"""Dataset statistics for the progress report: size, speakers, gender balance,
clip duration, and sentence length in characters and words.

Words are counted with khmer-nltk's word segmenter (Khmer has no spaces between words).
Sentence lengths use the cleaned training text (src/normalize.py), without spaces.

Usage:
    python src/dataset_stats.py      # writes results/dataset_stats.{csv,md} + results/fig_*.png
"""

import csv
import glob
import os
import statistics
import sys
import unicodedata
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from khmernltk import word_tokenize

sys.path.insert(0, os.path.dirname(__file__))
from normalize import clean_text  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FLEURS = f"{ROOT}/data/raw/fleurs_km"
SLR42 = f"{ROOT}/data/raw/slr42/km_kh_male"
OUT = f"{ROOT}/results"

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"FLEURS": "#2a78d6", "SLR42": "#eb6834"}


def n_words(text):
    tokens = word_tokenize(text.replace(" ", ""))
    return sum(1 for t in tokens if any(unicodedata.category(c)[0] in "LM" for c in t))


def load():
    csv.field_size_limit(sys.maxsize)
    rows = []
    for split in ("train", "dev", "test"):
        with open(f"{FLEURS}/{split}.tsv", encoding="utf-8", newline="") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                rows.append({"dataset": "FLEURS", "split": split, "sentence": r[0], "gender": r[6].lower(),
                             "speaker": None, "duration": int(r[5]) / 16000, "text": clean_text(r[2])})
    with open(f"{SLR42}/line_index.tsv", encoding="utf-8") as f:
        for line in f:
            utt, _, text = line.rstrip("\n").split("\t")
            with wave.open(f"{SLR42}/wavs/{utt}.wav") as w:
                dur = w.getnframes() / w.getframerate()
            rows.append({"dataset": "SLR42", "split": "train", "sentence": text, "gender": "male",
                         "speaker": utt.split("_")[1], "duration": dur,
                         "text": clean_text(text, remove_spaces=True)})
    for r in rows:
        r["chars"] = len(r["text"].replace(" ", ""))
        r["words"] = n_words(r["text"])
    return rows


def summarize(rows, name):
    d = [r["duration"] for r in rows]
    w = [r["words"] for r in rows]
    c = [r["chars"] for r in rows]
    male = sum(1 for r in rows if r["gender"] == "male")
    speakers = {r["speaker"] for r in rows if r["speaker"]}
    return {
        "subset": name, "clips": len(rows), "hours": round(sum(d) / 3600, 2),
        "unique_sentences": len({r["sentence"] for r in rows}),
        "speakers": len(speakers) if speakers else "n/a",
        "male_pct": round(100 * male / len(rows), 1),
        "dur_mean_s": round(statistics.mean(d), 1), "dur_median_s": round(statistics.median(d), 1),
        "dur_max_s": round(max(d), 1),
        "words_mean": round(statistics.mean(w), 1), "words_median": statistics.median(w), "words_max": max(w),
        "chars_mean": round(statistics.mean(c), 1), "chars_median": statistics.median(c),
    }


def style(ax, xlabel):
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel("Clips", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def histograms(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200)
    for ax, key, xlabel, bins in [
        (axes[0], "words", "Words per sentence (khmer-nltk)", range(0, 80, 3)),
        (axes[1], "duration", "Clip duration (seconds)", [x / 2 for x in range(0, 82)]),
    ]:
        for name in ("FLEURS", "SLR42"):
            vals = [r[key] for r in rows if r["dataset"] == name]
            ax.hist(vals, bins=bins, color=COLORS[name], alpha=0.75, label=name,
                    edgecolor="white", linewidth=0.6)
        style(ax, xlabel)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_data_distribution.png", bbox_inches="tight")
    print(f"wrote {OUT}/fig_data_distribution.png")


def main():
    rows = load()
    subsets = [
        ("FLEURS train", [r for r in rows if r["dataset"] == "FLEURS" and r["split"] == "train"]),
        ("FLEURS dev", [r for r in rows if r["dataset"] == "FLEURS" and r["split"] == "dev"]),
        ("FLEURS test", [r for r in rows if r["dataset"] == "FLEURS" and r["split"] == "test"]),
        ("SLR42 (all train)", [r for r in rows if r["dataset"] == "SLR42"]),
    ]
    stats = [summarize(r, n) for n, r in subsets]
    with open(f"{OUT}/dataset_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    keys = list(stats[0])
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(s[k]) for k in keys) + " |" for s in stats]
    open(f"{OUT}/dataset_stats.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    histograms(rows)


if __name__ == "__main__":
    main()
