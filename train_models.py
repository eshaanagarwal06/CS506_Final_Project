
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
        "RMSE": -scores["test_rmse"].mean(),
        "R2": scores["test_r2"].mean(),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nCross-validation results:")
print(results_df.to_string(index=False))

# 5. Select best model using CV MAE
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

best_model.fit(X_train, y_train)

# 6. Evaluate once on held-out test data
predictions = best_model.predict(X_test)

print(f"\nBest model: {best_name}")
print("Test MAE:",
      mean_absolute_error(y_test, predictions))
print("Test RMSE:",
      np.sqrt(mean_squared_error(y_test, predictions)))
print("Test R2:",
      r2_score(y_test, predictions))
