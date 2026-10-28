"""Text normalization for Khmer transcripts (PLAN.md, decision D3).

Two levels:
  clean_text()        -> training labels. Fixes noise but keeps the text readable.
  normalize_for_cer() -> scoring only. Strips everything that should not count as an error
                         (spaces, punctuation, digit style, Latin case).

Decided so far (see results/data_inspection.md for the evidence):
  - Unicode NFC
  - remove zero-width / invisible characters
  - remove Latin-only glosses in parentheses: "ចន នីសុន (John Nixon)" -> "ចន នីសុន"
  - SLR42 word spaces: both options kept (remove_spaces flag) to compare later
Pending until the Cultural Dataset is inspected:
  - punctuation in labels ("keep" / "strip")
  - digit style ("keep" / "khmer" / "ascii")
"""

import re
import unicodedata

INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
KHMER_TO_ASCII_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
ASCII_TO_KHMER_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")

# Parentheses with no Khmer inside and at least one Latin letter.
LATIN_GLOSS = re.compile(r"\s*\([^()ក-៿᧠-᧿]*[A-Za-z][^()ក-៿᧠-᧿]*\)")
WHITESPACE = re.compile(r"\s+")


def strip_punctuation(text):
    # Unicode category P* covers ។ ៕ ៖ and ASCII punctuation. ៗ (repetition sign) is a
    # letter modifier (Lm), not punctuation, so it is kept.
    return "".join(c for c in text if not unicodedata.category(c).startswith("P"))


def clean_text(text, remove_spaces=False, punctuation="keep", digits="keep"):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(INVISIBLE)
    text = LATIN_GLOSS.sub("", text)
    if punctuation == "strip":
        text = strip_punctuation(text)
    if digits == "khmer":
        text = text.translate(ASCII_TO_KHMER_DIGITS)
    elif digits == "ascii":
        text = text.translate(KHMER_TO_ASCII_DIGITS)
    text = WHITESPACE.sub("" if remove_spaces else " ", text)
    return text.strip()


def normalize_for_cer(text):
    text = clean_text(text, remove_spaces=True, punctuation="strip", digits="ascii")
    return text.lower()
