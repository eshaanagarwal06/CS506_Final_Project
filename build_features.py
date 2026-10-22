import re
from collections import Counter

import pandas as pd


# ========== SETTINGS ==========

INPUT_FILE = "dev-other_base.en_phonemes_complete.csv"
OUTPUT_FILE = "dev-other_base.en_features.csv"

# Number of most common within-word phoneme bigrams to keep as features
TOP_BIGRAMS = 20

# ARPAbet vowels (each vowel phoneme is one syllable nucleus)
VOWELS = set(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
)

CONSONANTS = set(
    "B CH D DH F G HH JH K L M N NG P R S SH T TH "
    "V W Y Z ZH".split()
)

PHONEMES = sorted(VOWELS | CONSONANTS)


# 1. Load phoneme results
df = pd.read_csv(INPUT_FILE, keep_default_na=False)

if "word_phonemes" not in df.columns:
    raise ValueError(
        f"{INPUT_FILE} has no word_phonemes column. "
        "Re-run complete_phonemes first."
    )


def split_words(word_phonemes):
    # "DH AH | K AE T" -> [["DH", "AH"], ["K", "AE", "T"]]
    return [
        word.split()
        for word in str(word_phonemes).split("|")
        if word.strip()
    ]


def consonant_clusters(phones):
    # Runs of 2+ consonants inside one word, e.g. "S T R" in "STREET"
    clusters = []
    run = []

    for phone in phones + [None]:
        if phone in CONSONANTS:
            run.append(phone)
        else:
            if len(run) >= 2:
                clusters.append(run)
            run = []

    return clusters


def word_bigrams(words):
    # Bigrams don't cross word boundaries
    return [
        f"{a}_{b}"
        for phones in words
        for a, b in zip(phones, phones[1:])
    ]


# 2. Pick the most common bigrams across all clips
# (uses no WER information, so it's safe to do before a train/test split)
bigram_totals = Counter()

for entry in df["word_phonemes"]:
    bigram_totals.update(word_bigrams(split_words(entry)))

top_bigrams = [
    bigram for bigram, _ in bigram_totals.most_common(TOP_BIGRAMS)
]


# 3. Build features for one clip
def build_features(row):

    words = split_words(row["word_phonemes"])
    phones = [phone for word in words for phone in word]

    text_words = re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        str(row["reference_text"]).lower()
    )

    num_words = len(words)
    num_phonemes = len(phones)
    num_vowels = sum(phone in VOWELS for phone in phones)

    clusters = [
        cluster
        for word in words
        for cluster in consonant_clusters(word)
    ]

    g2p_words = [
        word for word in row["g2p_words"].split(",")
        if word.strip()
    ]

    features = {
        "num_words": num_words,
        "avg_word_length": (
            sum(len(word) for word in text_words) / len(text_words)
            if text_words else 0
        ),
        "num_phonemes": num_phonemes,
        "phonemes_per_word": num_phonemes / num_words if num_words else 0,
        "num_syllables": num_vowels,
        "syllables_per_word": num_vowels / num_words if num_words else 0,
        "vowel_ratio": num_vowels / num_phonemes if num_phonemes else 0,
        "consonant_ratio": (
            (num_phonemes - num_vowels) / num_phonemes
            if num_phonemes else 0
        ),
        "num_consonant_clusters": len(clusters),
        "clusters_per_word": len(clusters) / num_words if num_words else 0,
        "max_cluster_length": max((len(c) for c in clusters), default=0),
        # Words missing from CMUdict are mostly names and rare words,
        # which Whisper tends to misspell regardless of their phonemes
        "num_g2p_words": len(g2p_words),
    }

    # Share of the clip's phonemes that are each phoneme
    counts = Counter(phones)

    for phoneme in PHONEMES:
        features[f"ph_{phoneme}"] = (
            counts[phoneme] / num_phonemes if num_phonemes else 0
        )

    # Share of the clip's bigrams that are each top bigram
    bigrams = word_bigrams(words)
    bigram_counts = Counter(bigrams)

    for bigram in top_bigrams:
        features[f"bi_{bigram}"] = (
            bigram_counts[bigram] / len(bigrams) if bigrams else 0
        )

    return pd.Series(features)


# 4. Apply to every clip and save
features = df.apply(build_features, axis=1)

output = pd.concat(
    [df[["audio_id", "speaker_id", "wer"]], features],
    axis=1
)

output.to_csv(OUTPUT_FILE, index=False)

print("\n--- FEATURES ---")
print("Total clips:", len(output))
print("Feature columns:", features.shape[1])
print("Top bigrams:", ", ".join(top_bigrams))

print("\nStrongest correlations with WER:")
correlations = (
    output.drop(columns=["audio_id", "speaker_id"])
    .corr(numeric_only=True)["wer"]
    .drop("wer")
    .dropna()
)
print(
    correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    ).head(10).to_string()
)

print(f"\nResults saved to {OUTPUT_FILE}")
