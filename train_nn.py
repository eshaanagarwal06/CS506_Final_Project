"""Train the neural-network (MLP) WER regressor independently.

Run from anywhere:
    python train_nn.py
    python train_nn.py --cv-folds 3  # quicker exploratory comparison

This is a feed-forward neural network implemented by scikit-learn's
MLPRegressor. It does NOT run in train_models.py. Both scripts must use the
same feature CSV and --cv-folds for directly comparable validation scores.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Keep these identical to train_models.py.
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "data"
DATA_CANDIDATES = (
    RESULTS_DIR / "train-other-500_base.en_features.csv.gz",
    RESULTS_DIR / "train-other-500_base.en_features.csv",
)
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

    # A feed-forward NN (MLP) with two hidden layers. Tune this separately.
    # The CPU-based scikit-learn version requires no extra ML dependencies.
    nn_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            batch_size=1024,       # Larger mini-batches for ~148k observations
            alpha=1.0,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    )

    cv = GroupKFold(n_splits=args.cv_folds)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    print(f"Cross-validating NN across {args.cv_folds} unseen-speaker folds...", flush=True)
    scores = cross_validate(
        nn_model, X_train, y_train, groups=groups_train,
        cv=cv, scoring=scoring, n_jobs=1,
    )
    cv_df = pd.DataFrame([{
        "Model": "Neural Network (MLP)",
        "MAE": -scores["test_mae"].mean(),
        "MAE_std": scores["test_mae"].std(),
        "RMSE": -scores["test_rmse"].mean(),
        "R2": scores["test_r2"].mean(),
    }])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_file = RESULTS_DIR / "nn_cv_scores.csv"
    cv_df.to_csv(cv_file, index=False)
    print("\nNeural network cross-validation results:")
    print(cv_df.to_string(index=False))

    # Fit on all training speakers once, and evaluate once on held-out speakers.
    print("\nFitting neural network on all training speakers...", flush=True)
    nn_model.fit(X_train, y_train)
    predictions = nn_model.predict(X_test)
    baseline_predictions = DummyRegressor(strategy="mean").fit(
        X_train, y_train
    ).predict(X_test)
    test_df = pd.DataFrame({
        "Neural Network (MLP)": evaluate(y_test, predictions),
        "Baseline": evaluate(y_test, baseline_predictions),
    }).T
    test_df.index.name = "Model"
    test_file = RESULTS_DIR / "nn_test_scores.csv"
    test_df.to_csv(test_file)
    print("\nHeld-out test:")
    print(test_df.to_string())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_test, predictions, alpha=0.13, s=5, color="#4f81bd")
    # Visualization range only; evaluation includes all held-out observations.
    limit = max(0.1, float(np.percentile(np.r_[y_test, predictions], 99.5)) * 1.05)
    ax.plot([0, limit], [0, limit], color="black", linewidth=0.8,
            label="Perfect prediction")
    ax.axhline(baseline_predictions[0], color="gray", linestyle="--",
               label="Mean baseline")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Actual WER")
    ax.set_ylabel("Predicted WER (NN)")
    ax.set_title(
        "NN: held-out unseen-speaker predictions\n"
        f"MAE {test_df.loc['Neural Network (MLP)', 'MAE']:.3f} vs. baseline "
        f"{test_df.loc['Baseline', 'MAE']:.3f}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    plot_file = RESULTS_DIR / "nn_model_performance.svg"
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {cv_file}, {test_file}, {plot_file}")


if __name__ == "__main__":
    main()
