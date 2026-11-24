# Analyzing and Predicting Speech Recognition Errors Using Phonetic Patterns
## Description:

Our project will investigate whether the phonetic composition of a spoken sentence can be used to predict the Word Error Rate (WER) of an automatic speech recognition system. We will use labeled speech datasets such as LibriSpeech, which provides audio clips paired with ground-truth transcripts, in addition to our own devised/sources transcripts and associated recorded speeches. Each audio clip will be processed through an ASR model, and the predicted transcript will be compared with the correct transcript to calculate WER.

The ground-truth transcripts will then be converted into phoneme sequences using the CMU Pronouncing Dictionary. From these sequences, we will extract sentence-level features such as phoneme frequencies, vowel and consonant proportions, average phonemes per word, consonant clusters, and selected phoneme combinations.

We will first perform exploratory analysis to identify relationships between phonetic features and transcription accuracy. We will then train regression models to predict WER from these features and evaluate how much of the variation in ASR performance can be explained by the phonetic composition of the sentence. If time permits, we will compare multiple ASR models to determine whether similar phonetic patterns lead to errors across different systems.


## How to Run

### Setup

Requires Python 3.10 or newer (we use 3.12).

```bash
git clone https://github.com/eshaanagarwal06/CS506_Final_Project.git
cd CS506_Final_Project

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng cmudict
```

The LibriSpeech audio is streamed from Hugging Face the first time you run the pipeline, so you don't need to download it separately.

### Pipeline

The project uses two LibriSpeech splits. `dev-other` (2,864 clips) is used for exploratory analysis. `train-other-500` (148,688 clips) is used to train and evaluate the models.

Run every command from the repository root. `prepare_data.py`, `build_features.py` and `visualize_data.py` read and write paths relative to the current folder. The training scripts find the `data/` folder on their own.

**1. Explore dev-other**

```bash
python prepare_data.py
python build_features.py
python visualize_data.py
```

**2. Train on train-other-500**

```bash
python phonemes.py data/train-other-500_base.en_results.csv data/train-other-500_base.en_processed.csv.gz
python build_features.py data/train-other-500_base.en_processed.csv.gz data/train-other-500_base.en_features.csv.gz
python train_models.py
python train_nn.py
python feature_ablation.py
```

| Step | What it does | Output |
| --- | --- | --- |
| `prepare_data.py` | Transcribes each clip with Whisper, computes WER against the LibriSpeech transcript (both normalized with Whisper's English normalizer), and converts the transcript to phonemes with CMUdict (falling back to G2P for words missing from CMUdict, mostly names), keeping word boundaries. The phoneme code lives in `phonemes.py` | `dev-other_base.en_processed.csv` |
| `phonemes.py` | Adds the phoneme columns to a results CSV that already has WER, so Whisper doesn't need to rerun | `data/train-other-500_base.en_processed.csv.gz` |
| `build_features.py` | Builds one row of phonetic features per clip, plus `wer`, and prints the 10 features most correlated with WER | `dev-other_base.en_features.csv`, `data/train-other-500_base.en_features.csv.gz` |
| `visualize_data.py` | Plots each feature's correlation with WER on dev-other, with the p < 0.05 significance band, and the distribution of WER | `phoneme_corr.svg`, `feature_corr.svg`, `wer_distribution.svg` |
| `train_models.py` | Compares a mean baseline, Ridge, Random Forest and Histogram Gradient Boosting with 5-fold cross-validation grouped by speaker, then scores the best model and the baseline once on a held-out set of unseen speakers | `data/classical_cv_scores.csv`, `data/classical_test_scores.csv`, `data/classical_model_performance.svg` |
| `train_nn.py` | Runs the same cross-validation and held-out test for a neural network (scikit-learn `MLPRegressor`). It is kept separate from `train_models.py` so it can be developed independently | `data/nn_cv_scores.csv`, `data/nn_test_scores.csv`, `data/nn_model_performance.svg` |
| `feature_ablation.py` | Greedy backward feature selection: drops one feature at a time while doing so significantly improves CV MAE, then compares the selected features with all features on the held-out speakers | `data/ablation_log.csv`, `data/ablation_candidates.csv`, `data/ablation_selected_features.txt`, `data/ablation_test_scores.csv`, `data/ablation_progress.svg` |

`prepare_data.py` is the slowest step. On the full dev-other split it took about 10 minutes on an M4 Max, and it will be much slower on older laptops. It saves a checkpoint (`*.partial.csv`) every 250 clips. The other dev-other steps take about 15 seconds in total. On train-other-500, `phonemes.py` and `build_features.py` take about 30 and 40 seconds on an M4 Max.

Both feature files are committed, so you can skip straight to `visualize_data.py` or to the training scripts without rerunning Whisper or rebuilding features. `data/train-other-500_base.en_results.csv` is not run through Whisper here: it already has Whisper transcripts and WER for all 148,688 clips.

Paths ending in `.gz` are compressed, because the uncompressed files would be over GitHub's 100 MB limit. The train-other-500 feature file (21 MB) is committed. The processed file (55 MB) is not, since it takes 30 seconds to rebuild.

### Training scripts

`train_models.py`, `train_nn.py` and `feature_ablation.py` use the same data, split and metrics, so their scores can be compared directly:

- **Features:** only the 71 phonetic columns (the scalar features below plus every `ph_` and `bi_` column). IDs, `wer`, Whisper outputs and timing columns are never used as inputs.
- **Split:** 20% of speakers are held out as the test set (`GroupShuffleSplit`, `random_state=42`). No speaker appears in both training and test data.
- **Validation:** cross-validation on the training speakers only, grouped by speaker (`GroupKFold`).
- **Metrics:** MAE, RMSE and R². Models are ranked by CV MAE. The held-out test set is scored once, after the model is chosen.
- **Baseline:** predicts the mean training WER for every clip.

`train_models.py` and `train_nn.py` keep `RANDOM_STATE`, `TEST_SIZE` and `DEFAULT_CV_FOLDS` at the top of each file, and these must stay identical. `feature_ablation.py` imports them, together with the data loader, from `train_models.py`, so keep the two files in the same folder.

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--data` | all three | `data/train-other-500_base.en_features.csv.gz`, else the uncompressed `.csv` | Feature file to train on |
| `--cv-folds` | all three | 5 (10 for `feature_ablation.py`) | Number of speaker-grouped CV folds. Use `--cv-folds 3` for a quicker run, and pass the same value to `train_models.py` and `train_nn.py` when comparing them |
| `--model` | `feature_ablation.py` | `hgb` | Model used to score features: `hgb` (a lighter Histogram Gradient Boosting) or `ridge` (much faster) |
| `--subsample-frac` | `feature_ablation.py` | 0.2 | Fraction of training speakers used during ablation. The final comparison uses the full training split |
| `--min-delta`, `--z` | `feature_ablation.py` | 0.0005, 1.0 | A removal counts only if it lowers mean CV MAE by at least `min-delta` and by more than `z` standard errors of the paired per-fold differences |
| `--max-rounds`, `--min-features` | `feature_ablation.py` | none, 1 | Optional limits on how many features are removed |
| `--jobs` | `feature_ablation.py` | -1 (all cores) | Parallel workers across candidate features |

For a fast ablation run, use `python feature_ablation.py --subsample-frac 0.1 --model ridge`.

### Settings

The Whisper settings are at the top of `prepare_data.py`:

- `DATASET_NAME`: which LibriSpeech split to use (`dev-clean`, `dev-other`, `test-clean`, `test-other`).
- `MODEL_NAME`: the Whisper model (for example `tiny.en`, `base.en`, `small.en`).
- `NUM_CLIPS`: how many clips to process. `None` processes the whole split; a small number like `20` is useful for a quick test.

`phonemes.py` and `build_features.py` take input and output paths as arguments (`build_features.py` defaults to the dev-other files). `visualize_data.py` is still hardcoded to the dev-other feature file; update `DATA_FILE` to plot another split. The training scripts take `--data` (see above).

The 20 bigram columns are the split's own most common bigrams, so they differ slightly between splits (`bi_AH_T` for train-other-500, `bi_AH_S` for dev-other). To give a split the same columns as another, pass `--bigrams-from` with that split's feature file, for example `--bigrams-from data/train-other-500_base.en_features.csv.gz`.

The train-other-500 results CSV also has `avg_logprob`, `audio_seconds` and `processing_seconds`. These come from Whisper or the audio, not the transcript, so `build_features.py` leaves them out.

### Feature columns

Both feature files have `audio_id`, `speaker_id`, `wer`, and these 71 features:

- **Length:** `num_words`, `avg_word_length` (letters), `num_phonemes`, `phonemes_per_word`, `num_syllables` (vowel phonemes), `syllables_per_word`
- **Vowels and consonants:** `vowel_ratio`, `consonant_ratio`
- **Consonant clusters** (2+ consonants in a row within one word): `num_consonant_clusters`, `clusters_per_word`, `max_cluster_length`
- **Out-of-dictionary words:** `num_g2p_words`, the number of words missing from CMUdict
- **Phonemes:** `ph_<PHONEME>`, the share of the clip's phonemes that are that phoneme (39 columns)
- **Bigrams:** `bi_<A>_<B>`, the share of the clip's within-word phoneme bigrams that are that bigram, for the 20 most common bigrams in the split

## CSV summary

Each data CSV has one row per audio clip. The transcription files use Whisper Base English (`base.en`).

| File | LibriSpeech split | Contents | Clips | Speakers | Corpus WER | Mean clip WER |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [`train-other-500_base.en_results.csv`](data/train-other-500_base.en_results.csv) | `train-other-500` | Transcripts, errors, confidence, and timing | 148,688 | 1,166 | 6.50% | 7.47% |
| [`train-other-500_base.en_features.csv.gz`](data/train-other-500_base.en_features.csv.gz) | `train-other-500` | Model input with 71 features, used by the training scripts | 148,688 | 1,166 | 6.50% | 7.47% |
| [`dev-other_base.en_processed.csv`](dev-other_base.en_processed.csv) | `dev-other` | Current transcripts and phonemes | 2,864 | 33 | 10.40% | 12.12% |
| [`dev-other_base.en_features.csv`](dev-other_base.en_features.csv) | `dev-other` | 71 features, used by `visualize_data.py` | 2,864 | 33 | 10.40% | 12.12% |
| [`dev_other_results_base.en.csv`](dev_other_results_base.en.csv) | `dev-other` | Earlier transcription export | 2,864 | 33 | 10.44% | 12.17% |
| [`dev-other_base.en_phonemes.csv`](dev-other_base.en_phonemes.csv) | `dev-other` | Earlier CMUdict phoneme export | 2,864 | 33 | 10.44% | 12.17% |
| [`dev-other_base.en_phonemes_complete.csv`](dev-other_base.en_phonemes_complete.csv) | `dev-other` | Earlier phoneme export with G2P fallback | 2,864 | 33 | 10.44% | 12.17% |

Corpus WER divides all word errors by all reference words. Mean clip WER gives every clip equal weight. The CSVs store `wer` as a fraction (for example, `0.10` means 10%).

The feature files carry over the WER values from the files they were built from. All five `dev-other` files cover the same clips. Ten clips have different WER values between the earlier exports and the current pipeline outputs.

The processed files add `phonemes`, `word_phonemes` (the same phonemes with ` | ` between words), `num_phonemes`, `g2p_words` and `unresolved_words` to the transcription columns.

The `train-other-500` CSV records 315,674 errors across 4,858,957 normalized reference words. Reference transcripts come from the [LibriSpeech ASR Corpus](https://www.openslr.org/12/) by Panayotov, Chen, Povey, and Khudanpur, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Model output files

The training scripts write these to `data/`. Each has one row per model or feature set.

| File | Written by | Contents |
| --- | --- | --- |
| `classical_cv_scores.csv` | `train_models.py` | CV MAE, MAE standard deviation across folds, RMSE and R² for the baseline and each classical model |
| `classical_test_scores.csv` | `train_models.py` | MAE, RMSE and R² on the held-out speakers for the best classical model and the baseline |
| `nn_cv_scores.csv` | `train_nn.py` | The same CV scores for the neural network |
| `nn_test_scores.csv` | `train_nn.py` | Held-out scores for the neural network and the baseline |
| `ablation_log.csv` | `feature_ablation.py` | One row per removal round: the feature removed, features left, CV MAE and its standard deviation, and the gain and its standard error |
| `ablation_candidates.csv` | `feature_ablation.py` | CV MAE without each candidate feature, for every round |
| `ablation_selected_features.txt` | `feature_ablation.py` | The selected features, one per line |
| `ablation_test_scores.csv` | `feature_ablation.py` | Held-out scores using all features and using the selected features |

## Rough Timeline
### 🔵 Mid October
- Finalize the dataset and ASR model.
- Build the initial data-processing pipeline.
- Generate ASR transcripts and calculate WER for each audio clip.
- Convert ground-truth transcripts into phoneme sequences using CMUdict.
- Complete phoneme-based feature engineering.
- Perform exploratory data analysis and create visualizations showing relationships between phonetic features and WER.
### 🔵 Mid November
- Train baseline and more advanced regression models.
- Begin evaluating model performance and identifying important features.
- Finalize model evaluation and interpretation.
### 🔵 End of Semester
- Compare results across models or ASR systems if time permits.
- Complete final visualizations and summarize findings and limitations.
- Clean the GitHub repository and prepare the final report and presentation.

## Goals:

  Automatic speech recognition systems can perform very differently depending on the words and sounds contained in an audio sample. Our project will investigate whether the phonetic composition of a spoken sentence can be used to predict how accurately an automatic speech recognition (ASR) model will transcribe it.

Our main research question is:

Can we predict the Word Error Rate (WER) of an audio clip using features derived from the phoneme composition of its ground-truth transcript?  Success means our model beats a baseline model that predicts the average WER for all training clips in the dataset, measured by mean squared error (MSE) or mean absolute error (MAE).

Rather than predicting whether each individual word is transcribed correctly, we will treat each audio clip or sentence as one observation. For every clip, we will calculate the actual WER produced by an ASR system and extract phoneme-based features from the correct transcript. We will then train regression models to predict WER from these features.

A secondary goal will be to determine whether certain phonemes or phoneme combinations are associated with higher transcription error rates.


## Data Collection:

 We plan to use an existing labeled speech dataset such as LibriSpeech, which contains English audio recordings paired with verified ground-truth transcripts. Using an existing speech corpus will allow us to analyze a large and reproducible set of recordings without needing to collect our own audio.

For each audio clip, we will:
- Obtain the original audio and its ground-truth transcript.
- Process the audio using one or more automatic speech recognition models.
- Compare the predicted transcript with the ground-truth transcript.
- Calculate the Word Error Rate (WER) for the clip.
- Convert the ground-truth transcript into phoneme sequences using the CMU Pronouncing Dictionary (CMUdict).

Word Error Rate will be calculated as:

$$
WER = \frac{S + D + I}{N}
$$

where (S) represents substitutions, (D) represents deletions, (I) represents insertions, and (N) is the number of words in the reference transcript.

We will get and process this data using the Hugging Face datasets library, which provides a convenient interface for downloading and working with speech datasets.


## Feature Extraction

Each audio clip will be represented using features describing the phonetic composition of its transcript.

### Possible features include:

- Frequency of each phoneme
- Proportion of vowels and consonants
- Total number of phonemes
- Average number of phonemes per word
- Number of syllables
- Frequency of consonant clusters
- Frequency of selected phoneme combinations or bigrams
- Sentence length
- Average word length

For example, a sentence containing many occurrences of phonemes such as TH, R, or difficult consonant clusters may have a different transcription error rate from a sentence composed primarily of simpler sound combinations.

## Visualization

Before training models, we will examine how transcription errors vary with different phonetic characteristics.

### Potential visualizations include:

- Distribution of WER across audio clips
- Average WER for sentences containing different phonemes
- Correlation between phoneme frequency and WER
- Heatmaps showing WER associated with different phoneme combinations
- Relationship between sentence length and WER
- Comparison of phoneme-related error patterns across ASR models


## Resources

- [LibriSpeech ASR Corpus](https://www.openslr.org/12/) – Speech dataset containing audio recordings paired with ground-truth transcripts.
- [CMU Pronouncing Dictionary (CMUdict)](https://github.com/cmusphinx/cmudict) – English pronunciation dictionary used to convert words into phoneme sequences.
- [JiWER](https://github.com/jitsi/jiwer) – Python package for calculating Word Error Rate (WER) and other ASR evaluation metrics.
- [OpenAI Whisper](https://github.com/openai/whisper) – Automatic speech recognition model that can be used to generate predicted transcripts.
- [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets) – Large crowdsourced speech dataset that could be used as an extension for testing more diverse speakers and accents.
- [G2P (Grapheme-to-Phoneme) – g2p-en](https://github.com/Kyubyong/g2p?utm_source=chatgpt.com) – Python package used to convert text into phoneme sequences, complementing CMUdict by predicting pronunciations for words not found in the dictionary.
