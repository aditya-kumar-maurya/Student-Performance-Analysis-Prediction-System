
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 500

first_names = ["Aarav","Priya","Rahul","Neha","Riya","Aman","Anjali","Vivek","Sneha","Karan"]
last_names = ["Sharma","Kumar","Singh","Verma","Gupta","Yadav","Maurya","Mishra"]
departments = ["CSE","IT","ECE","ME","CE"]

df = pd.DataFrame({
    "student_id": [f"STU{1001+i}" for i in range(n)],
    "student_name": [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)],
    "age": rng.integers(18, 24, n),
    "gender": rng.choice(["Male","Female"], n),
    "department": rng.choice(departments, n, p=[.28,.20,.18,.18,.16]),
    "semester": rng.integers(1, 9, n),
    "attendance_percentage": np.clip(rng.normal(78,11,n),45,100).round(1),
    "study_hours_per_day": np.clip(rng.normal(4.5,1.7,n),1,10).round(1),
    "previous_exam_score": np.clip(rng.normal(68,14,n),25,98).round(1),
    "midterm_score": np.clip(rng.normal(70,13,n),25,98).round(1),
    "assignment_percentage": np.clip(rng.normal(74,12,n),30,100).round(1),
    "practical_lab_score": np.clip(rng.normal(76,12,n),30,100).round(1),
    "sleep_hours_per_day": np.clip(rng.normal(7,1.1,n),4,10).round(1),
    "online_learning_hours": np.clip(rng.normal(5,2.5,n),0,14).round(1),
    "previous_backlogs": np.clip(rng.poisson(.7,n),0,5),
    "internet_access": rng.choice(["Yes","No"],n,p=[.88,.12]),
    "quiz_score": np.clip(rng.normal(72,13,n),25,100).round(1),
    "class_participation_score": np.clip(rng.normal(70,15,n),20,100).round(1)
})

dept_bonus = df["department"].map({"CSE":3,"IT":2,"ECE":1,"ME":0,"CE":0}).astype(float)
internet_bonus = (df["internet_access"] == "Yes") * 1.2
df["final_score"] = np.clip(
    0.24*df["previous_exam_score"] +
    0.19*df["midterm_score"] +
    0.15*df["assignment_percentage"] +
    0.13*df["practical_lab_score"] +
    0.10*df["quiz_score"] +
    0.07*df["attendance_percentage"] +
    0.045*df["study_hours_per_day"]*10 +
    0.035*df["class_participation_score"] +
    0.02*df["online_learning_hours"]*10 +
    0.01*df["sleep_hours_per_day"]*10 -
    1.6*df["previous_backlogs"] + dept_bonus + internet_bonus +
    rng.normal(0,1.2,n), 0, 100
).round(1)

# Add a few missing values to demonstrate the cleaning step.
raw = df.copy()
for col in ["attendance_percentage","study_hours_per_day","previous_exam_score",
            "midterm_score","assignment_percentage","practical_lab_score",
            "sleep_hours_per_day","online_learning_hours","quiz_score",
            "class_participation_score"]:
    for idx in rng.choice(n, 3, replace=False):
        raw.loc[idx, col] = np.nan

for col in ["gender","department","internet_access"]:
    for idx in rng.choice(n, 2, replace=False):
        raw.loc[idx, col] = np.nan

raw.to_csv(OUT/"student_performance_raw.csv", index=False)
print("Created 500 synthetic student records.")
