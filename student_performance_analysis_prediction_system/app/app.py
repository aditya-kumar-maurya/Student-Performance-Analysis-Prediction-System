from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "student_performance.db"
MODEL_PATH = BASE_DIR / "model" / "student_performance_model.joblib"
CSV_PATH = BASE_DIR / "data" / "student_performance_500.csv"

app = Flask(__name__)
app.secret_key = "student-performance-demo-secret-key"

MODEL_FEATURES = [
    "age", "gender", "department", "semester", "attendance_percentage",
    "study_hours_per_day", "previous_exam_score", "midterm_score",
    "assignment_percentage", "practical_lab_score", "sleep_hours_per_day",
    "online_learning_hours", "previous_backlogs", "internet_access", "quiz_score"
]

STUDENT_COLUMNS = [
    "student_id", "student_name", "age", "gender", "department", "semester",
    "attendance_percentage", "study_hours_per_day", "previous_exam_score",
    "midterm_score", "assignment_percentage", "practical_lab_score",
    "sleep_hours_per_day", "online_learning_hours", "previous_backlogs",
    "internet_access", "quiz_score", "class_participation_score", "final_score"
]

NUMERIC_COLUMNS = [
    "age", "semester", "attendance_percentage", "study_hours_per_day",
    "previous_exam_score", "midterm_score", "assignment_percentage",
    "practical_lab_score", "sleep_hours_per_day", "online_learning_hours",
    "previous_backlogs", "quiz_score", "class_participation_score", "final_score"
]

model = joblib.load(MODEL_PATH)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def category(score):
    score = float(score)
    if score < 50:
        return "At Risk"
    if score < 60:
        return "Average"
    if score < 75:
        return "Good"
    return "Excellent"


def dashboard_data():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) AS n FROM students").fetchone()["n"]
    avg = conn.execute("SELECT AVG(final_score) AS v FROM students").fetchone()["v"] or 0
    attendance = conn.execute("SELECT AVG(attendance_percentage) AS v FROM students").fetchone()["v"] or 0
    at_risk = conn.execute("SELECT COUNT(*) AS n FROM students WHERE final_score < 60").fetchone()["n"]
    recent = conn.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 8").fetchall()
    students = conn.execute(
        "SELECT student_id, student_name, department, semester, attendance_percentage, "
        "previous_exam_score, final_score FROM students ORDER BY student_id LIMIT 12"
    ).fetchall()
    conn.close()
    return {"total": total, "avg": round(avg, 1), "attendance": round(attendance, 1),
            "at_risk": at_risk, "recent": recent, "students": students}


def form_to_student_data(form):
    data = {}
    for col in STUDENT_COLUMNS:
        value = form.get(col, "").strip()
        if col in NUMERIC_COLUMNS:
            if value == "":
                data[col] = None
            elif col in {"age", "semester", "previous_backlogs"}:
                data[col] = int(float(value))
            else:
                data[col] = float(value)
        else:
            data[col] = value
    return data


def validate_student(data, require_id=True):
    errors = []
    required_text = ["student_id", "student_name", "gender", "department", "internet_access"]
    if require_id:
        for col in required_text:
            if not str(data.get(col, "")).strip():
                errors.append(f"{col.replace('_', ' ').title()} is required.")
    for col in NUMERIC_COLUMNS:
        if data.get(col) is None:
            if col != "final_score":
                errors.append(f"{col.replace('_', ' ').title()} is required.")
    ranges = {
        "age": (16, 60), "semester": (1, 8), "attendance_percentage": (0, 100),
        "study_hours_per_day": (0, 24), "previous_exam_score": (0, 100),
        "midterm_score": (0, 100), "assignment_percentage": (0, 100),
        "practical_lab_score": (0, 100), "sleep_hours_per_day": (0, 24),
        "online_learning_hours": (0, 24), "previous_backlogs": (0, 20),
        "quiz_score": (0, 100), "class_participation_score": (0, 100),
        "final_score": (0, 100),
    }
    for col, (lo, hi) in ranges.items():
        val = data.get(col)
        if val is not None and not (lo <= float(val) <= hi):
            errors.append(f"{col.replace('_', ' ').title()} must be between {lo} and {hi}.")
    return errors


@app.route("/")
def index():
    return render_template("index.html", **dashboard_data())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = {}
        for col in MODEL_FEATURES:
            value = request.form.get(col, "").strip()
            if col in ["gender", "department", "internet_access"]:
                payload[col] = value
            elif col in ["age", "semester", "previous_backlogs"]:
                payload[col] = int(float(value))
            else:
                payload[col] = float(value)
        input_df = pd.DataFrame([payload], columns=MODEL_FEATURES)
        predicted = float(np.clip(model.predict(input_df)[0], 0, 100))
        perf = category(predicted)
        student_id = request.form.get("student_id", "").strip() or "GUEST"
        conn = get_db()
        conn.execute(
            "INSERT INTO prediction_history (student_id, predicted_score, performance_category, model_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (student_id, round(predicted, 1), perf, "Random Forest Regression", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit(); conn.close()
        return render_template("prediction_result.html", predicted_score=round(predicted, 1), performance=perf, student_id=student_id)
    except Exception as exc:
        flash(f"Prediction error: {exc}", "danger")
        return redirect(url_for("index") + "#prediction")


@app.route("/student/<student_id>")
def student(student_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    history = conn.execute("SELECT * FROM prediction_history WHERE student_id = ? ORDER BY id DESC LIMIT 10", (student_id,)).fetchall()
    conn.close()
    if not row:
        flash("Student ID not found.", "warning")
        return redirect(url_for("index") + "#student-analysis")
    return render_template("student.html", student=row, history=history, category=category)


@app.route("/students")
def students():
    q = request.args.get("q", "").strip()
    conn = get_db()
    if q:
        rows = conn.execute(
            "SELECT * FROM students WHERE student_id LIKE ? OR student_name LIKE ? OR department LIKE ? ORDER BY student_id",
            (f"%{q}%", f"%{q}%", f"%{q}%")
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM students ORDER BY student_id").fetchall()
    conn.close()
    return render_template("students.html", students=rows, q=q)


@app.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("student_form.html", mode="add", student=None)
    try:
        data = form_to_student_data(request.form)
        errors = validate_student(data)
        if errors:
            for error in errors: flash(error, "danger")
            return render_template("student_form.html", mode="add", student=data)
        conn = get_db()
        conn.execute(
            "INSERT INTO students (" + ",".join(STUDENT_COLUMNS) + ") VALUES (" + ",".join(["?"] * len(STUDENT_COLUMNS)) + ")",
            [data[c] for c in STUDENT_COLUMNS],
        )
        conn.commit(); conn.close()
        flash(f"Student {data['student_id']} added successfully to SQLite.", "success")
        return redirect(url_for("students"))
    except sqlite3.IntegrityError as exc:
        flash(f"Could not add student. Student ID may already exist. {exc}", "danger")
        return render_template("student_form.html", mode="add", student=data)
    except Exception as exc:
        flash(f"Add student error: {exc}", "danger")
        return render_template("student_form.html", mode="add", student=request.form)


@app.route("/students/<student_id>/edit", methods=["GET", "POST"])
def edit_student(student_id):
    conn = get_db()
    existing = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()
    if not existing:
        flash("Student not found.", "warning")
        return redirect(url_for("students"))
    if request.method == "GET":
        return render_template("student_form.html", mode="edit", student=existing)
    try:
        data = form_to_student_data(request.form)
        data["student_id"] = student_id
        errors = validate_student(data)
        if errors:
            for error in errors: flash(error, "danger")
            return render_template("student_form.html", mode="edit", student=data)
        update_cols = [c for c in STUDENT_COLUMNS if c != "student_id"]
        conn = get_db()
        conn.execute(
            "UPDATE students SET " + ", ".join(f"{c} = ?" for c in update_cols) + " WHERE student_id = ?",
            [data[c] for c in update_cols] + [student_id],
        )
        conn.commit(); conn.close()
        flash(f"Student {student_id} updated successfully using SQLite UPDATE.", "success")
        return redirect(url_for("students"))
    except Exception as exc:
        flash(f"Update error: {exc}", "danger")
        return render_template("student_form.html", mode="edit", student=data)


@app.post("/students/<student_id>/delete")
def delete_student(student_id):
    conn = get_db()
    cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit(); conn.close()
    if cur.rowcount:
        flash(f"Student {student_id} deleted from SQLite.", "success")
    else:
        flash("Student not found.", "warning")
    return redirect(url_for("students"))


@app.route("/students/import", methods=["GET", "POST"])
def import_csv():
    if request.method == "GET":
        return render_template("import_csv.html")
    uploaded = request.files.get("csv_file")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        flash("Please select a CSV file.", "danger")
        return redirect(url_for("import_csv"))
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in STUDENT_COLUMNS if c not in df.columns]
        if missing:
            flash("CSV is missing columns: " + ", ".join(missing), "danger")
            return redirect(url_for("import_csv"))
        df = df[STUDENT_COLUMNS].copy()
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[c for c in STUDENT_COLUMNS if c != "final_score"])
        df["final_score"] = df["final_score"].fillna(0)
        records = [tuple(row[c] for c in STUDENT_COLUMNS) for _, row in df.iterrows()]
        conn = get_db()
        conn.executemany(
            "INSERT OR REPLACE INTO students (" + ",".join(STUDENT_COLUMNS) + ") VALUES (" + ",".join(["?"] * len(STUDENT_COLUMNS)) + ")",
            records,
        )
        conn.commit(); conn.close()
        flash(f"Imported {len(records)} student records into SQLite using Pandas + SQL.", "success")
        return redirect(url_for("students"))
    except Exception as exc:
        flash(f"CSV import error: {exc}", "danger")
        return redirect(url_for("import_csv"))


@app.route("/history")
def history():
    conn = get_db()

    rows = conn.execute(
        "SELECT * FROM prediction_history ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=rows
    )


@app.route("/history/view/<int:prediction_id>")
def view_prediction(prediction_id):
    conn = get_db()

    prediction = conn.execute(
        "SELECT * FROM prediction_history WHERE id = ?",
        (prediction_id,)
    ).fetchone()

    conn.close()

    if not prediction:
        flash("Prediction record not found.", "warning")
        return redirect(url_for("history"))

    return render_template(
        "prediction_detail.html",
        prediction=prediction
    )


@app.post("/history/delete/<int:prediction_id>")
def delete_prediction(prediction_id):
    conn = get_db()

    cur = conn.execute(
        "DELETE FROM prediction_history WHERE id = ?",
        (prediction_id,)
    )

    conn.commit()
    conn.close()

    if cur.rowcount > 0:
        flash(
            "Prediction record deleted successfully.",
            "success"
        )
    else:
        flash(
            "Prediction record not found.",
            "warning"
        )

    return redirect(url_for("history"))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "database": "SQLite", "model": "Random Forest Regression"})


if __name__ == "__main__":
    app.run(debug=True)