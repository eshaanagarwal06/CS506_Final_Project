"""Train classical WER regressors on LibriSpeech train-other-500 features.

Run from anywhere:
    python train_models.py
    python train_models.py --cv-folds 3  # quicker exploratory comparison

The neural network is intentionally in train_nn.py so two people can work
independently. Both scripts use the same data, random seed, speaker-grouped
train/test split, CV folds, and metrics. They save different output files.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Both training scripts must use the same settings for a fair comparison.
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "data"
DATA_CANDIDATES = (
    RESULTS_DIR / "train-other-500_base.en_features.csv.gz",
    RESULTS_DIR / "train-other-500_base.en_features.csv",
)

# Use only reference-text/phonetic features built by build_features.py.
# Do not accidentally include ASR predictions, error counts, or WER-derived columns.
SCALAR_FEATURES = {
    "num_words", "avg_word_length", "num_phonemes", "phonemes_per_word",
    "num_syllables", "syllables_per_word", "vowel_ratio", "consonant_ratio",
    "num_consonant_clusters", "clusters_per_word", "max_cluster_length",
    "num_g2p_words",
}


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}\n"
            "Run build_features.py on the train-other-500 processed file first."
        )
    df = pd.read_csv(path, dtype={"speaker_id": "string", "audio_id": "string"})
    required = {"wer", "speaker_id"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["speaker_id"].isna().any():
        raise ValueError("Missing speaker IDs; speaker-grouped splitting requires all IDs.")
    y = pd.to_numeric(df["wer"], errors="raise")
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("WER has NaN or infinite values; inspect the dataset.")
    columns = [
        col for col in df.columns
        if col in SCALAR_FEATURES or col.startswith(("ph_", "bi_"))
    ]
    if not columns:
        raise ValueError("No phonetic features found. Run build_features.py first.")
    X = df[columns].apply(pd.to_numeric, errors="raise")
    X = X.replace([np.inf, -np.inf], np.nan)
    groups = df["speaker_id"].astype(str)
    return X, y, groups


def evaluate(y_true, predicted):
    return {
        "MAE": mean_absolute_error(y_true, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_true, predicted)),
        "R2": r2_score(y_true, predicted),
    }


def plot_results(cv_df, y_test, predictions, baseline_predictions, best_name, test_df, output, folds):
    fig, (ax_cv, ax_pred) = plt.subplots(1, 2, figsize=(14, 5.5))
    ordered = cv_df.sort_values("MAE", ascending=False)
    for i, row in enumerate(ordered.itertuples()):
        ax_cv.errorbar(
            row.MAE, i, xerr=row.MAE_std, fmt="o", capsize=4, markersize=8,
            color="gray" if row.Model == "Baseline" else "#4f81bd",
        )
    ax_cv.set_yticks(range(len(ordered)), ordered["Model"])
    baseline_mae = cv_df.loc[cv_df["Model"] == "Baseline", "MAE"].iloc[0]
    ax_cv.axvline(baseline_mae, color="black", linestyle="--", label="Mean baseline")
    ax_cv.set_xlabel("CV MAE (lower is better; whiskers: +/- 1 fold SD)")
    ax_cv.set_title(f"Classical models: {folds}-fold speaker-grouped CV")
    ax_cv.legend()

    ax_pred.scatter(y_test, predictions, alpha=0.13, s=5, color="#4f81bd")
    # Scale for readability on ~148k observations; metrics still use ALL test rows.
    limit = max(0.1, float(np.percentile(np.r_[y_test, predictions], 99.5)) * 1.05)
    ax_pred.plot([0, limit], [0, limit], color="black", linewidth=0.8,
                 label="Perfect prediction")
    ax_pred.axhline(baseline_predictions[0], color="gray", linestyle="--",
                    label="Mean baseline")
    ax_pred.set_xlim(0, limit)
    ax_pred.set_ylim(0, limit)
    ax_pred.set_xlabel("Actual WER")
    ax_pred.set_ylabel(f"Predicted WER ({best_name})")
    ax_pred.set_title(
        f"Unseen-speaker test: {best_name}\n"
        f"MAE {test_df.loc[best_name, 'MAE']:.3f} vs. baseline "
        f"{test_df.loc['Baseline', 'MAE']:.3f}"
    )
    ax_pred.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Feature CSV or CSV.gz")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    data_path = args.data if args.data else next(
        (p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0]
    )
    X, y, groups = load_data(data_path)
    if groups.nunique() < args.cv_folds + 1:
        raise ValueError("Not enough speakers for the requested number of CV folds.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    print(f"Features: {data_path}")
    print(f"Total: {len(X):,} clips, {groups.nunique():,} speakers, {X.shape[1]} features")
    print(f"Train: {len(train_idx):,} clips; held-out test: {len(test_idx):,} clips "
          f"from {groups.iloc[test_idx].nunique():,} unseen speakers")

    models = {
        "Baseline": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=100,        # Reasonable starting point for 148k clips
                min_samples_leaf=10,
                max_depth=16,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "Histogram Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv = GroupKFold(n_splits=args.cv_folds)
    rows = []
    for name, model in models.items():
        print(f"Cross-validating {name}...", flush=True)
        scores = cross_validate(
            model, X_train, y_train, groups=groups_train,
            cv=cv, scoring=scoring, n_jobs=1,
        )
        rows.append({
            "Model": name,
            "MAE": -scores["test_mae"].mean(),
            "MAE_std": scores["test_mae"].std(),
            "RMSE": -scores["test_rmse"].mean(),
            "R2": scores["test_r2"].mean(),
        })

    cv_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    cv_file = RESULTS_DIR / "classical_cv_scores.csv"
    cv_df.to_csv(cv_file, index=False)
    print("\nClassical model cross-validation results:")
    print(cv_df.to_string(index=False))

    # Choose using training-only CV, then inspect the held-out test ONCE.
    best_name = cv_df.iloc[0]["Model"]
    best_model = models[best_name].fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    baseline_predictions = models["Baseline"].fit(X_train, y_train).predict(X_test)
    test_df = pd.DataFrame({
        best_name: evaluate(y_test, predictions),
        "Baseline": evaluate(y_test, baseline_predictions),
    }).T
    test_df.index.name = "Model"
    test_file = RESULTS_DIR / "classical_test_scores.csv"
    test_df.to_csv(test_file)
    print(f"\nHeld-out test: best classical model = {best_name}")
    print(test_df.to_string())

    plot_file = RESULTS_DIR / "classical_model_performance.svg"
    plot_results(cv_df, y_test, predictions, baseline_predictions,
                 best_name, test_df, plot_file, args.cv_folds)
    print(f"\nSaved: {cv_file}, {test_file}, {plot_file}")


if __name__ == "__main__":
    main()
