import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TRAIN_URL = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
TEST_URL = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
LOCAL_TRAIN = "/mnt/data/assignment3.csv"
LOCAL_TEST = "/mnt/data/assignment3test.csv"


def _read_csv_with_fallback(primary: str, fallback: str) -> pd.DataFrame:
    try:
        return pd.read_csv(primary)
    except Exception:
        return pd.read_csv(fallback)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
    data["hour"] = data["DateTime"].dt.hour.fillna(0).astype(int)
    data["minute"] = data["DateTime"].dt.minute.fillna(0).astype(int)
    data["dayofweek"] = data["DateTime"].dt.dayofweek.fillna(0).astype(int)
    data["month"] = data["DateTime"].dt.month.fillna(0).astype(int)
    data["day"] = data["DateTime"].dt.day.fillna(0).astype(int)
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    drop_cols = [c for c in ["id", "DateTime", "meal"] if c in data.columns]
    data = data.drop(columns=drop_cols)

    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.fillna(0)


train = _read_csv_with_fallback(TRAIN_URL, LOCAL_TRAIN)
test = _read_csv_with_fallback(TEST_URL, LOCAL_TEST)

X_train = _prepare_features(train)
y_train = train["meal"].astype(int)
X_test = _prepare_features(test)

shared_cols = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[shared_cols]
X_test = X_test[shared_cols]

# Required forecasting algorithm
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced"
)

# Required fitted model
modelFit = model.fit(X_train, y_train)

# Required predictions for all 1000 test observations
pred = [int(x) for x in modelFit.predict(X_test)]
