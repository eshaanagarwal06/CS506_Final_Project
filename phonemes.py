"""Convert reference transcripts into ARPAbet phonemes.

Uses CMUdict where it has the word and falls back to G2P for words it is
missing (mostly names and rare words).

prepare_data.py imports add_phonemes() after running Whisper. To add phonemes
to a results CSV that already has WER (so Whisper doesn't need to rerun), use:

    python phonemes.py INPUT_RESULTS_CSV OUTPUT_PROCESSED_CSV

Output paths ending in .gz are compressed automatically.
"""

import argparse
import re
from functools import lru_cache

import cmudict
import pandas as pd
from g2p_en import G2p

# Valid ARPAbet phonemes
VALID_PHONEMES = set(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH "
    "IH IY JH K L M N NG OW OY P R S SH T TH "
    "UH UW V W Y Z ZH".split()
)

WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Loaded on first use, since both take a few seconds
_cmu = None
_g2p = None


@lru_cache(maxsize=None)
def pronounce_word(word):
    global _cmu, _g2p

    if _cmu is None:
        _cmu = cmudict.dict()

    if word in _cmu:
        phones = _cmu[word][0]
        source = "cmudict"

    else:
        if _g2p is None:
            _g2p = G2p()

        predicted = _g2p(word)

        phones = [
            phone for phone in predicted
            if isinstance(phone, str)
            and re.sub(r"\d", "", phone) in VALID_PHONEMES
        ]

        source = "g2p" if phones else "unresolved"

    # Drop stress markers: "AH0" -> "AH"
    phones = [re.sub(r"\d", "", phone) for phone in phones]

    return phones, source


def extract_phonemes(text):
    words = WORD_PATTERN.findall(str(text).lower())

    phonemes = []
    word_phonemes = []
    g2p_words = []
    unresolved_words = []

    for word in words:
        phones, source = pronounce_word(word)

        phonemes.extend(phones)
        word_phonemes.append(" ".join(phones))

        if source == "g2p":
            g2p_words.append(word)
        elif source == "unresolved":
            unresolved_words.append(word)

    return {
        "phonemes": " ".join(phonemes),
        # Same phonemes with word boundaries kept, e.g. "DH AH | K AE T"
        "word_phonemes": " | ".join(word_phonemes),
        "num_phonemes": len(phonemes),
        "g2p_words": ", ".join(g2p_words),
        "unresolved_words": ", ".join(unresolved_words),
    }


def add_phonemes(df):
    """Return a copy of df with phoneme columns built from reference_text."""
    phoneme_columns = pd.DataFrame(
        [extract_phonemes(text) for text in df["reference_text"]],
        index=df.index,
    )

    df = df.drop(columns=phoneme_columns.columns, errors="ignore")
    return pd.concat([df, phoneme_columns], axis=1)


def print_summary(df):
    print("\n--- PHONEME COMPLETION ---")
    print("Total clips:", len(df))
    print("Clips using G2P:", (df["g2p_words"] != "").sum())
    print("Clips with unresolved words:", (df["unresolved_words"] != "").sum())

    vocabulary = {
        word.strip()
        for entry in df["g2p_words"] if entry
        for word in entry.split(",")
    }
    print("Unique words pronounced with G2P:", len(vocabulary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add phoneme columns to a Whisper results CSV."
    )
    parser.add_argument("input", help="results CSV with a reference_text column")
    parser.add_argument("output", help="where to save the processed CSV")
    args = parser.parse_args()

    results = pd.read_csv(args.input, keep_default_na=False)
    processed = add_phonemes(results)
    processed.to_csv(args.output, index=False)

    print_summary(processed)
    print(f"\nResults saved to {args.output}")
