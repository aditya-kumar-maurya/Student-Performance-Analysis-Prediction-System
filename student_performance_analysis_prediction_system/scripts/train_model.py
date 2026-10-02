
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "student_performance_raw.csv"
CLEAN = ROOT / "data" / "student_performance_500.csv"
MODEL = ROOT / "model" / "student_performance_model.joblib"

df = pd.read_csv(RAW)

# Data cleaning with Pandas.
numeric_cols = [
    "age","semester","attendance_percentage","study_hours_per_day",
    "previous_exam_score","midterm_score","assignment_percentage",
    "practical_lab_score","sleep_hours_per_day","online_learning_hours",
    "previous_backlogs","quiz_score","class_participation_score","final_score"
]
categorical_cols = ["gender","department","internet_access"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df["student_name"] = df["student_name"].fillna("Unknown Student")
df.to_csv(CLEAN, index=False)

# Exactly 15 prediction features.
features = [
    "age","gender","department","semester","attendance_percentage",
    "study_hours_per_day","previous_exam_score","midterm_score",
    "assignment_percentage","practical_lab_score","sleep_hours_per_day",
    "online_learning_hours","previous_backlogs","internet_access","quiz_score"
]
target = "final_score"

X = df[features]
y = df[target]

numeric = [c for c in features if c not in categorical_cols]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), ["gender","department","internet_access"])
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3))
print("R2  :", round(r2_score(y_test, pred), 3))

joblib.dump(pipeline, MODEL)
print("Saved:", MODEL)
