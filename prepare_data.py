import io
import re

import pandas as pd
import soundfile as sf
import whisper
from whisper.normalizers import EnglishTextNormalizer
from datasets import load_dataset, Audio
from jiwer import wer
from itertools import islice

from phonemes import add_phonemes, print_summary


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

# 6. Add phonemes from the reference transcripts (see phonemes.py)
df = add_phonemes(df)

# 7. Save results
df.to_csv(OUTPUT_FILE, index=False)

print_summary(df)
print(f"\nResults saved to {OUTPUT_FILE}")
