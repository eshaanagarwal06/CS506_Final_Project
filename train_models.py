
import matplotlib
matplotlib.use("Agg")  # write files only; works on any OS without a display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

RANDOM_STATE = 42
DATA_FILE = "dev-other_base.en_features.csv"

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

y = df["wer"]
groups = df["speaker_id"]

X = (
    df.drop(columns=["wer", "audio_id", "speaker_id"])
    .select_dtypes(include="number")
)

# 2. Split by speaker
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]

# 3. Define models
models = {
    "Baseline": DummyRegressor(strategy="mean"),

    "Ridge": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10),
    ),

    "MLP": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=1.0,
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),

    "Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    ),
}

# 4. Cross-validation
cv = GroupKFold(n_splits=5)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring=scoring,
    )

    results.append({
        "Model": name,
        "MAE": -scores["test_mae"].mean(),
        "MAE_std": scores["test_mae"].std(),
        "RMSE": -scores["test_rmse"].mean(),
        "R2": scores["test_r2"].mean(),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nCross-validation results:")
print(results_df.drop(columns="MAE_std").to_string(index=False))

# 5. Select best model using CV MAE
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

best_model.fit(X_train, y_train)

# 6. Evaluate once on held-out test data, next to the mean baseline
predictions = best_model.predict(X_test)

baseline_model = models["Baseline"].fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)


def test_scores(pred):
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }


test_df = pd.DataFrame({
    best_name: test_scores(predictions),
    "Baseline": test_scores(baseline_predictions),
}).T

print(
    f"\nHeld-out test set: {len(test_idx)} clips from "
    f"{groups.iloc[test_idx].nunique()} unseen speakers "
    f"(small, so these numbers are noisy)"
)
print(test_df.to_string())

# 7. Plot model performance
fig, (ax_cv, ax_pred) = plt.subplots(1, 2, figsize=(14, 5.5))

ordered = results_df.sort_values("MAE", ascending=False)
for row_number, row in enumerate(ordered.itertuples()):
    ax_cv.errorbar(row.MAE, row_number, xerr=row.MAE_std, fmt="o",
                   color="gray" if row.Model == "Baseline" else "#4f81bd",
                   capsize=4, markersize=8)
ax_cv.set_yticks(range(len(ordered)), ordered["Model"])
baseline_mae = results_df.loc[results_df["Model"] == "Baseline", "MAE"].iloc[0]
ax_cv.axvline(baseline_mae, color="black", linestyle="--",
              label=f"Mean baseline ({baseline_mae:.4f})")
ax_cv.set_xlabel("Mean absolute error (lower is better; whiskers = ±1 SD across folds)")
ax_cv.set_title("5-fold speaker-grouped CV error by model")
ax_cv.legend(loc="lower right")

ax_pred.scatter(y_test, predictions, alpha=0.3, s=12, color="#4f81bd")
limit = max(y_test.max(), predictions.max()) * 1.05
ax_pred.plot([0, limit], [0, limit], color="black", linewidth=0.8,
             label="Perfect prediction")
ax_pred.axhline(baseline_predictions[0], color="gray", linestyle="--",
                label="Mean baseline")
ax_pred.set_xlim(0, limit)
ax_pred.set_ylim(0, limit)
ax_pred.set_xlabel("Actual WER")
ax_pred.set_ylabel(f"Predicted WER ({best_name})")
ax_pred.set_title(
    f"Held-out test set: {best_name} predicted vs. actual\n"
    f"MAE {test_df.loc[best_name, 'MAE']:.3f} vs. baseline "
    f"{test_df.loc['Baseline', 'MAE']:.3f}"
)
ax_pred.legend(loc="upper left")

fig.tight_layout()
fig.savefig("model_performance.svg", bbox_inches="tight")
plt.close(fig)
print("\nSaved model_performance.svg")
