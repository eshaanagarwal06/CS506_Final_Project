"""Greedy backward feature ablation for the WER regressor.

Procedure (training data only; the held-out test speakers are never used to
choose features):

  1. Take a speaker-grouped subsample of the training split (fast iteration).
  2. Score the current feature set with speaker-grouped CV (MAE).
  3. For every remaining feature, drop it and re-score.
  4. If the best drop improves CV MAE significantly, remove that feature and
     go back to step 3 starting from the reduced set.
  5. Stop when no single removal gives a significant improvement (converged).
  6. Refit on the FULL training split with the selected features vs. all
     features and evaluate once on the held-out unseen-speaker test set.

"Significant" means the mean per-fold MAE improvement (same folds for both
sets, so the comparison is paired) is at least --min-delta AND at least
--z standard errors of the paired fold differences.

Run from the same folder as train_models.py:
    python feature_ablation.py
    python feature_ablation.py --subsample-frac 0.1 --model ridge   # much faster
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the exact data loading, seed, split size, and metrics from train_models.py
from train_models import (
    DATA_CANDIDATES,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    evaluate,
    load_data,
)

ABLATION_CV_FOLDS = 3


def make_model(name):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)
        )
    # Lighter than the full HGB in train_models.py, since it is fit thousands of times.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    )


def fold_scores(model, Xn, yn, splits, col_idx):
    """Per-fold MAE using only the columns in col_idx."""
    Xs = Xn[:, col_idx]
    maes = []
    for tr, te in splits:
        m = clone(model).fit(Xs[tr], yn[tr])
        maes.append(mean_absolute_error(yn[te], m.predict(Xs[te])))
    return np.array(maes)


def subsample_speakers(groups, frac, min_speakers, seed):
    """Boolean mask selecting a random fraction of SPEAKERS (never splits a speaker)."""
    speakers = np.array(sorted(groups.unique()))
    n = min(len(speakers), max(min_speakers, int(round(len(speakers) * frac))))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(speakers, size=n, replace=False)
    return groups.isin(chosen).to_numpy()


def plot_progress(log_df, output, min_delta):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["n_features"], log_df["cv_mae"], marker="o", color="#4f81bd")
    ax.fill_between(
        log_df["n_features"],
        log_df["cv_mae"] - log_df["cv_mae_std"],
        log_df["cv_mae"] + log_df["cv_mae_std"],
        alpha=0.2, color="#4f81bd",
    )
    ax.invert_xaxis()
    ax.set_xlabel("Number of features remaining")
    ax.set_ylabel("CV MAE on subsample (lower is better)")
    ax.set_title(f"Backward ablation (stop when gain < {min_delta} or not significant)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="Feature CSV or CSV.gz")
    parser.add_argument("--model", choices=["hgb", "ridge"], default="hgb",
                        help="Model used for ablation scoring (ridge is far faster)")
    parser.add_argument("--subsample-frac", type=float, default=0.2,
                        help="Fraction of training SPEAKERS used during ablation")
    parser.add_argument("--cv-folds", type=int, default=ABLATION_CV_FOLDS)
    parser.add_argument("--min-delta", type=float, default=0.0005,
                        help="Minimum mean CV-MAE improvement to count as a change")
    parser.add_argument("--z", type=float, default=1.0,
                        help="Improvement must also exceed z * SE of paired fold differences")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Optional cap on the number of removals")
    parser.add_argument("--min-features", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel workers across candidate features")
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not 0 < args.subsample_frac <= 1:
        parser.error("--subsample-frac must be in (0, 1]")

    data_path = args.data if args.data else next(
        (p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0]
    )
    X, y, groups = load_data(data_path)

    # Same speaker-grouped split as train_models.py / train_nn.py.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # Smaller, speaker-grouped slice of the TRAINING data for ablation only.
    mask = subsample_speakers(groups_train, args.subsample_frac,
                              args.cv_folds + 1, RANDOM_STATE)
    X_sub, y_sub, g_sub = X_train[mask], y_train[mask], groups_train[mask]
    Xn = X_sub.to_numpy(dtype=float)
    yn = y_sub.to_numpy(dtype=float)
    splits = list(GroupKFold(n_splits=args.cv_folds).split(Xn, yn, g_sub))
    names = list(X.columns)

    print(f"Features: {data_path}")
    print(f"Ablation subsample: {len(X_sub):,} clips from {g_sub.nunique():,} speakers "
          f"(of {len(X_train):,} training clips)")
    print(f"Starting features: {len(names)} | model: {args.model} | "
          f"{args.cv_folds}-fold grouped CV\n")

    model = make_model(args.model)
    current = list(range(len(names)))
    cur_scores = fold_scores(model, Xn, yn, splits, current)

    log_rows = [{
        "round": 0, "removed_feature": "(none)", "n_features": len(current),
        "cv_mae": cur_scores.mean(), "cv_mae_std": cur_scores.std(),
        "gain": np.nan, "gain_se": np.nan,
    }]
    candidate_rows = []
    print(f"Round 0: {len(current)} features, CV MAE {cur_scores.mean():.5f}")

    round_no = 0
    converged = False
    while len(current) > args.min_features:
        if args.max_rounds is not None and round_no >= args.max_rounds:
            print(f"Stopped: reached --max-rounds ({args.max_rounds}).")
            break
        round_no += 1
        start = time.time()

        results = Parallel(n_jobs=args.jobs)(
            delayed(fold_scores)(model, Xn, yn, splits, [c for c in current if c != f])
            for f in current
        )
        means = np.array([r.mean() for r in results])
        for f, r in zip(current, results):
            candidate_rows.append({"round": round_no, "feature": names[f],
                                   "cv_mae_without": r.mean()})

        best = int(np.argmin(means))
        diff = cur_scores - results[best]          # positive = removal helped
        gain = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        significant = gain >= args.min_delta and gain > args.z * se

        if not significant:
            print(f"Round {round_no}: best removal '{names[current[best]]}' changes MAE by "
                  f"{gain:+.5f} (SE {se:.5f}) -> not significant. Converged.")
            converged = True
            break

        removed = current.pop(best)
        cur_scores = results[best]
        log_rows.append({
            "round": round_no, "removed_feature": names[removed],
            "n_features": len(current), "cv_mae": cur_scores.mean(),
            "cv_mae_std": cur_scores.std(), "gain": gain, "gain_se": se,
        })
        print(f"Round {round_no}: removed '{names[removed]}' | gain {gain:+.5f} "
              f"(SE {se:.5f}) | {len(current)} features left | "
              f"CV MAE {cur_scores.mean():.5f} | {time.time() - start:.0f}s", flush=True)

    if not converged and len(current) <= args.min_features:
        print("Stopped: reached --min-features.")

    selected = [names[c] for c in current]
    print(f"\nSelected {len(selected)} of {len(names)} features.")

    # Final check: full training split, held-out unseen speakers, evaluated once each.
    final_model = make_model(args.model)
    all_pred = clone(final_model).fit(X_train, y_train).predict(X_test)
    sel_pred = clone(final_model).fit(X_train[selected], y_train).predict(X_test[selected])
    test_df = pd.DataFrame({
        f"All features ({len(names)})": evaluate(y_test, all_pred),
        f"Selected features ({len(selected)})": evaluate(y_test, sel_pred),
    }).T
    test_df.index.name = "Feature set"
    print("\nHeld-out test (full training data):")
    print(test_df.to_string())

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    log_df = pd.DataFrame(log_rows)
    log_file = RESULTS_DIR / "ablation_log.csv"
    cand_file = RESULTS_DIR / "ablation_candidates.csv"
    sel_file = RESULTS_DIR / "ablation_selected_features.txt"
    test_file = RESULTS_DIR / "ablation_test_scores.csv"
    plot_file = RESULTS_DIR / "ablation_progress.svg"

    log_df.to_csv(log_file, index=False)
    pd.DataFrame(candidate_rows).to_csv(cand_file, index=False)
    sel_file.write_text("\n".join(selected) + "\n")
    test_df.to_csv(test_file)
    plot_progress(log_df, plot_file, args.min_delta)
    print(f"\nSaved: {log_file}, {cand_file}, {sel_file}, {test_file}, {plot_file}")


if __name__ == "__main__":
    main()
