"""Character-level inspection of Khmer transcripts.

Used to decide the text normalization rules (PLAN.md, decision D3):
which digits, punctuation, whitespace and non-Khmer characters each source uses.

Usage (TSV sources, column index is 0-based):
    python src/inspect_text.py fleurs_raw=data/raw/fleurs_km/train.tsv:2 \
                               slr42=data/raw/slr42/line_index.tsv:1 \
                               -o results/data_inspection.md

From other code (e.g. a Kaggle notebook):
    from inspect_text import report
    print(report(list_of_texts, "cultural"))
"""

import argparse
import csv
import statistics
import sys
import unicodedata
from collections import Counter

KHMER_DIGITS = set("០១២៣៤៥៦៧៨៩")
KHMER_PUNCT = set("។៕៖ៗ៘៙៚")
WHITESPACE_NAMES = {
    " ": "SPACE",
    "​": "ZERO WIDTH SPACE",
    "‌": "ZERO WIDTH NON-JOINER",
    "‍": "ZERO WIDTH JOINER",
    " ": "NO-BREAK SPACE",
    "\t": "TAB",
    "﻿": "BYTE ORDER MARK",
}


def char_class(c):
    if c in KHMER_DIGITS:
        return "khmer_digit"
    if c.isascii() and c.isdigit():
        return "ascii_digit"
    if c in KHMER_PUNCT:
        return "khmer_punct"
    if c in WHITESPACE_NAMES:
        return "whitespace"
    if "ក" <= c <= "៿" or "᧠" <= c <= "᧿":
        return "khmer_letter"
    if c.isascii() and c.isalpha():
        return "latin_letter"
    if unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S"):
        return "other_punct_symbol"
    return "other"


def char_name(c):
    return WHITESPACE_NAMES.get(c) or unicodedata.name(c, f"U+{ord(c):04X}")


def report(texts, name, max_examples=5):
    texts = [t for t in texts if t is not None]
    n = len(texts)
    lengths = [len(t) for t in texts]
    chars = Counter(c for t in texts for c in t)
    classes = Counter()
    for c, k in chars.items():
        classes[char_class(c)] += k

    def frac(pred):
        return sum(1 for t in texts if pred(t)) / n if n else 0.0

    lines = [f"## {name}", ""]
    lines.append(f"- Transcripts: **{n}**")
    if n:
        lines.append(
            f"- Length (chars): mean {statistics.mean(lengths):.0f}, "
            f"median {statistics.median(lengths):.0f}, max {max(lengths)}"
        )
    lines.append(f"- Change under Unicode NFC: {frac(lambda t: unicodedata.normalize('NFC', t) != t):.1%}")
    lines.append(f"- Ends with ។ or ៕: {frac(lambda t: t.rstrip()[-1:] in ('។', '៕')):.1%}")
    lines.append(f"- Contains any Khmer punctuation: {frac(lambda t: any(c in KHMER_PUNCT for c in t)):.1%}")
    lines.append(f"- Contains a space: {frac(lambda t: ' ' in t):.1%}")
    lines.append(f"- Contains a zero-width space: {frac(lambda t: chr(0x200B) in t):.1%}")
    lines.append(f"- Contains Khmer digits: {frac(lambda t: any(c in KHMER_DIGITS for c in t)):.1%}")
    lines.append(f"- Contains ASCII digits: {frac(lambda t: any(c.isascii() and c.isdigit() for c in t)):.1%}")
    lines.append(f"- Contains Latin letters: {frac(lambda t: any(c.isascii() and c.isalpha() for c in t)):.1%}")
    lines.append("")

    lines.append("| Character class | Count |")
    lines.append("|---|---|")
    for k, v in classes.most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    non_letters = [(c, k) for c, k in chars.most_common() if char_class(c) != "khmer_letter"]
    if non_letters:
        lines.append("| Non-letter character | Unicode name | Count |")
        lines.append("|---|---|---|")
        for c, k in non_letters[:40]:
            shown = repr(c) if char_class(c) == "whitespace" else c
            lines.append(f"| {shown} | {char_name(c)} | {k} |")
        lines.append("")

    for label, pred in [
        ("ASCII digits", lambda t: any(c.isascii() and c.isdigit() for c in t)),
        ("Latin letters", lambda t: any(c.isascii() and c.isalpha() for c in t)),
        ("other characters", lambda t: any(char_class(c) in ("other", "other_punct_symbol") for c in t)),
    ]:
        examples = [t for t in texts if pred(t)][:max_examples]
        if examples:
            lines.append(f"Examples with {label}:")
            lines.extend(f"- `{t}`" for t in examples)
            lines.append("")
    return "\n".join(lines)


def read_tsv_column(path, col):
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", newline="") as f:
        return [row[col] for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(row) > col]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="name=path.tsv:column (repeat path with commas to merge files)")
    parser.add_argument("-o", "--output", help="write the markdown report here instead of stdout")
    args = parser.parse_args()

    sections = ["# Transcript inspection report", ""]
    for spec in args.sources:
        name, rest = spec.split("=", 1)
        paths, col = rest.rsplit(":", 1)
        texts = [t for p in paths.split(",") for t in read_tsv_column(p, int(col))]
        sections.append(report(texts, name))

    out = "\n".join(sections)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
