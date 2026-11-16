import io
import re

import pandas as pd
import soundfile as sf
import whisper
from whisper.normalizers import EnglishTextNormalizer
from datasets import load_dataset, Audio
from jiwer import wer
from itertools import islice
import cmudict
from g2p_en import G2p
from functools import lru_cache


# ========== EXPERIMENT SETTINGS ==========

DATASET_NAME = "dev-other"
MODEL_NAME = "base.en"
NUM_CLIPS = None  # None = full split
OUTPUT_FILE = f"{DATASET_NAME}_{MODEL_NAME}_processed.csv"

# Available LibriSpeech datasets
DATASET_OPTIONS = {
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test")
}


# 1. Load the selected LibriSpeech dataset

if DATASET_NAME not in DATASET_OPTIONS:
    raise ValueError(f"Invalid dataset: {DATASET_NAME}")

config, split = DATASET_OPTIONS[DATASET_NAME]

dataset = load_dataset(
    "openslr/librispeech_asr",
    config,
    split=split,
    streaming=True
)

dataset = dataset.cast_column(
    "audio", Audio(decode=False)
)

print(f"Using dataset: {DATASET_NAME}")

# 2. Load Whisper once
model = whisper.load_model(MODEL_NAME)

# 3. Normalize text
# Whisper's normalizer makes "FIFTEEN" and "15" (or "MR" and "Mr.") compare equal
english_normalizer = EnglishTextNormalizer()

def normalize(text):
    # LibriSpeech has no hyphens ("RELISTED"), so join Whisper's "re-listed"
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", text)
    return english_normalizer(text)

# 4. Process recordings
results = []

for i, sample in enumerate(islice(dataset, NUM_CLIPS)):

    reference = sample["text"]

    audio_info = sample["audio"]
    audio_source = (
        io.BytesIO(audio_info["bytes"])
        if audio_info["bytes"] is not None
        else audio_info["path"]
    )

    audio, sample_rate = sf.read(
        audio_source,
        dtype="float32"
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != 16000:
        raise ValueError("Expected 16 kHz audio")

    prediction = model.transcribe(
        audio,
        language="en",
        fp16=False
    )["text"]

    error = wer(
        normalize(reference),
        normalize(prediction)
    )

    results.append({
        "audio_id": sample["id"],
        "speaker_id": sample["speaker_id"],
        "reference_text": reference,
        "predicted_text": prediction,
        "wer": error
    })

    print(
        f"Clip {i+1}/{NUM_CLIPS or 'all'} | "
        f"WER: {error:.2%}"
    )

    # Checkpoint so a long run isn't lost if it crashes
    if (i + 1) % 250 == 0:
        pd.DataFrame(results).to_csv(
            f"{DATASET_NAME}_{MODEL_NAME}.partial.csv",
            index=False
        )

# 5. Summarize results
df = pd.DataFrame(results)

print("\n--- SUMMARY ---")
print("Total clips:", len(df))
print("Average clip WER:", df["wer"].mean())
print("Clips with errors:", (df["wer"] > 0).sum())

# 6. Extract complete phonemes from reference transcripts

# Load pronunciation tools
cmu = cmudict.dict()
g2p = G2p()

# Valid ARPAbet phonemes
VALID_PHONEMES = set(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH "
    "IH IY JH K L M N NG OW OY P R S SH T TH "
    "UH UW V W Y Z ZH".split()
)


@lru_cache(maxsize=None)
def pronounce_word(word):

    if word in cmu:
        phones = cmu[word][0]
        source = "cmudict"

    else:
        predicted = g2p(word)

        phones = [
            phone for phone in predicted
            if isinstance(phone, str)
            and re.sub(r"\d", "", phone) in VALID_PHONEMES
        ]

        source = "g2p" if phones else "unresolved"

    phones = [
        re.sub(r"\d", "", phone)
        for phone in phones
    ]

    return phones, source


def extract_phonemes(text):

    words = re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        str(text).lower()
    )

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

    return pd.Series({
        "phonemes": " ".join(phonemes),
        # Same phonemes with word boundaries kept, e.g. "DH AH | K AE T"
        "word_phonemes": " | ".join(word_phonemes),
        "num_phonemes": len(phonemes),
        "g2p_words": ", ".join(g2p_words),
        "unresolved_words": ", ".join(unresolved_words)
    })


# Extract phonemes for all transcripts
new_columns = df["reference_text"].apply(extract_phonemes)

for column in new_columns.columns:
    df[column] = new_columns[column]

# 7. Save results
df.to_csv(OUTPUT_FILE, index=False)

print("\n--- PHONEME COMPLETION ---")

print("Total clips:", len(df))

print(
    "Clips using G2P:",
    (df["g2p_words"] != "").sum()
)

print(
    "Clips with unresolved words:",
    (df["unresolved_words"] != "").sum()
)

print("\n--- G2P PRONUNCIATIONS ---")

g2p_vocabulary = set()

for entry in df["g2p_words"]:
    if entry:
        g2p_vocabulary.update(
            word.strip() for word in entry.split(",")
        )

for word in sorted(g2p_vocabulary):
    phones, _ = pronounce_word(word)
    print(f"{word}: {' '.join(phones)}")

print(f"\nResults saved to {OUTPUT_FILE}")
