from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "student_performance_model.joblib"
CSV_PATH = BASE_DIR / "data" / "student_performance_500.csv"


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = "student-performance-demo-secret-key"


# =========================================================
# MODEL FEATURES
# =========================================================

MODEL_FEATURES = [
    "age",
    "gender",
    "department",
    "semester",
    "attendance_percentage",
    "study_hours_per_day",
    "previous_exam_score",
    "midterm_score",
    "assignment_percentage",
    "practical_lab_score",
    "sleep_hours_per_day",
    "online_learning_hours",
    "previous_backlogs",
    "internet_access",
    "quiz_score"
]


# =========================================================
# STUDENT COLUMNS
# =========================================================

STUDENT_COLUMNS = [
    "student_id",
    "student_name",
    "age",
    "gender",
    "department",
    "semester",
    "attendance_percentage",
    "study_hours_per_day",
    "previous_exam_score",
    "midterm_score",
    "assignment_percentage",
    "practical_lab_score",
    "sleep_hours_per_day",
    "online_learning_hours",
    "previous_backlogs",
    "internet_access",
    "quiz_score",
    "class_participation_score",
    "final_score"
]


# =========================================================
# NUMERIC COLUMNS
# =========================================================

NUMERIC_COLUMNS = [
    "age",
    "semester",
    "attendance_percentage",
    "study_hours_per_day",
    "previous_exam_score",
    "midterm_score",
    "assignment_percentage",
    "practical_lab_score",
    "sleep_hours_per_day",
    "online_learning_hours",
    "previous_backlogs",
    "quiz_score",
    "class_participation_score",
    "final_score"
]


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# POSTGRESQL CONNECTION WRAPPER
# =========================================================

class PostgresConnection:

    def __init__(self, url):
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Please add DATABASE_URL to your .env file."
            )

        self.conn = psycopg2.connect(
            url,
            sslmode="require"
        )

    def execute(self, sql, params=None):

        # Existing project queries use ? placeholders.
        # PostgreSQL uses %s.
        sql = sql.replace("?", "%s")

        cursor = self.conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            sql,
            params or ()
        )

        return cursor

    def executemany(self, sql, seq_of_params):

        sql = sql.replace("?", "%s")

        cursor = self.conn.cursor()

        cursor.executemany(
            sql,
            seq_of_params
        )

        return cursor

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db():
    return PostgresConnection(DATABASE_URL)


# =========================================================
# CREATE POSTGRESQL TABLES
# =========================================================

def init_postgres():

    conn = get_db()

    try:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (

                student_id TEXT PRIMARY KEY,

                student_name TEXT NOT NULL,

                age INTEGER NOT NULL,

                gender TEXT NOT NULL,

                department TEXT NOT NULL,

                semester INTEGER NOT NULL,

                attendance_percentage DOUBLE PRECISION NOT NULL,

                study_hours_per_day DOUBLE PRECISION NOT NULL,

                previous_exam_score DOUBLE PRECISION NOT NULL,

                midterm_score DOUBLE PRECISION NOT NULL,

                assignment_percentage DOUBLE PRECISION NOT NULL,

                practical_lab_score DOUBLE PRECISION NOT NULL,

                sleep_hours_per_day DOUBLE PRECISION NOT NULL,

                online_learning_hours DOUBLE PRECISION NOT NULL,

                previous_backlogs INTEGER NOT NULL,

                internet_access TEXT NOT NULL,

                quiz_score DOUBLE PRECISION NOT NULL,

                class_participation_score DOUBLE PRECISION NOT NULL,

                final_score DOUBLE PRECISION

            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (

                id SERIAL PRIMARY KEY,

                student_id TEXT NOT NULL,

                predicted_score DOUBLE PRECISION NOT NULL,

                performance_category TEXT NOT NULL,

                model_name TEXT NOT NULL,

                created_at TEXT NOT NULL

            )
        """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# =========================================================
# CONVERT NUMPY VALUES TO PYTHON VALUES
# =========================================================

def python_value(value):

    if hasattr(value, "item"):
        return value.item()

    return value


# =========================================================
# SEED STUDENTS FROM CSV
# =========================================================

def seed_postgres_from_csv():

    if not CSV_PATH.exists():
        return

    conn = get_db()

    try:

        result = conn.execute(
            "SELECT COUNT(*) AS n FROM students"
        ).fetchone()

        count = result["n"]

        # If students already exist, do not import again.
        if count > 0:
            return

        df = pd.read_csv(CSV_PATH)

        missing = [
            column
            for column in STUDENT_COLUMNS
            if column not in df.columns
        ]

        if missing:
            print(
                "CSV is missing columns:",
                ", ".join(missing)
            )
            return

        df = df[STUDENT_COLUMNS].copy()

        # Convert numeric columns
        for column in NUMERIC_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Remove invalid rows
        df = df.dropna(
            subset=[
                column
                for column in STUDENT_COLUMNS
                if column != "final_score"
            ]
        )

        # If final score is missing, use 0
        df["final_score"] = df["final_score"].fillna(0)

        records = []

        for _, row in df.iterrows():

            record = tuple(
                python_value(row[column])
                for column in STUDENT_COLUMNS
            )

            records.append(record)

        placeholders = ",".join(
            ["%s"] * len(STUDENT_COLUMNS)
        )

        columns = ",".join(STUDENT_COLUMNS)

        sql = f"""
            INSERT INTO students ({columns})
            VALUES ({placeholders})
            ON CONFLICT (student_id)
            DO NOTHING
        """

        conn.executemany(
            sql,
            records
        )

        conn.commit()

        print(
            f"Successfully seeded {len(records)} students."
        )

    except Exception as exc:

        conn.rollback()

        print(
            "CSV seed error:",
            exc
        )

    finally:

        conn.close()


# =========================================================
# INITIALIZE DATABASE
# =========================================================

try:

    init_postgres()

    seed_postgres_from_csv()

except Exception as exc:

    print(
        "Database initialization error:",
        exc
    )


# =========================================================
# PERFORMANCE CATEGORY
# =========================================================

def category(score):

    score = float(score)

    if score < 50:
        return "At Risk"

    if score < 60:
        return "Average"

    if score < 75:
        return "Good"

    return "Excellent"


# =========================================================
# DASHBOARD DATA
# =========================================================

def dashboard_data():

    conn = get_db()

    try:

        total = conn.execute(
            "SELECT COUNT(*) AS n FROM students"
        ).fetchone()["n"]

        avg = conn.execute(
            "SELECT AVG(final_score) AS v FROM students"
        ).fetchone()["v"] or 0

        attendance = conn.execute(
            "SELECT AVG(attendance_percentage) AS v FROM students"
        ).fetchone()["v"] or 0

        at_risk = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM students
            WHERE final_score < 60
            """
        ).fetchone()["n"]

        recent = conn.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT 8
            """
        ).fetchall()

        students = conn.execute(
            """
            SELECT
                student_id,
                student_name,
                department,
                semester,
                attendance_percentage,
                previous_exam_score,
                final_score
            FROM students
            ORDER BY student_id
            LIMIT 12
            """
        ).fetchall()

        return {
            "total": total,
            "avg": round(avg, 1),
            "attendance": round(attendance, 1),
            "at_risk": at_risk,
            "recent": recent,
            "students": students
        }

    finally:

        conn.close()


# =========================================================
# FORM TO STUDENT DATA
# =========================================================

def form_to_student_data(form):

    data = {}

    for col in STUDENT_COLUMNS:

        value = form.get(
            col,
            ""
        ).strip()

        if col in NUMERIC_COLUMNS:

            if value == "":

                data[col] = None

            elif col in {
                "age",
                "semester",
                "previous_backlogs"
            }:

                data[col] = int(
                    float(value)
                )

            else:

                data[col] = float(value)

        else:

            data[col] = value

    return data


# =========================================================
# VALIDATE STUDENT
# =========================================================

def validate_student(data, require_id=True):

    errors = []

    required_text = [
        "student_id",
        "student_name",
        "gender",
        "department",
        "internet_access"
    ]

    if require_id:

        for col in required_text:

            if not str(
                data.get(col, "")
            ).strip():

                errors.append(
                    f"{col.replace('_', ' ').title()} is required."
                )

    for col in NUMERIC_COLUMNS:

        if data.get(col) is None:

            if col != "final_score":

                errors.append(
                    f"{col.replace('_', ' ').title()} is required."
                )

    ranges = {

        "age": (16, 60),

        "semester": (1, 8),

        "attendance_percentage": (0, 100),

        "study_hours_per_day": (0, 24),

        "previous_exam_score": (0, 100),

        "midterm_score": (0, 100),

        "assignment_percentage": (0, 100),

        "practical_lab_score": (0, 100),

        "sleep_hours_per_day": (0, 24),

        "online_learning_hours": (0, 24),

        "previous_backlogs": (0, 20),

        "quiz_score": (0, 100),

        "class_participation_score": (0, 100),

        "final_score": (0, 100),
    }

    for col, (lo, hi) in ranges.items():

        val = data.get(col)

        if val is not None:

            if not (
                lo <= float(val) <= hi
            ):

                errors.append(
                    f"{col.replace('_', ' ').title()} "
                    f"must be between {lo} and {hi}."
                )

    return errors


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        **dashboard_data()
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        payload = {}

        for col in MODEL_FEATURES:

            value = request.form.get(
                col,
                ""
            ).strip()

            if col in [
                "gender",
                "department",
                "internet_access"
            ]:

                payload[col] = value

            elif col in [
                "age",
                "semester",
                "previous_backlogs"
            ]:

                payload[col] = int(
                    float(value)
                )

            else:

                payload[col] = float(value)

        # Student ID is required
        student_id = request.form.get(
            "student_id",
            ""
        ).strip()

        if not student_id:

            raise ValueError(
                "Student ID is required. "
                "Please enter a valid Student ID."
            )

        # Check whether student exists
        conn = get_db()

        student_exists = conn.execute(
            """
            SELECT student_id
            FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()

        conn.close()

        if not student_exists:

            raise ValueError(
                f"Student ID {student_id} does not exist."
            )

        input_df = pd.DataFrame(
            [payload],
            columns=MODEL_FEATURES
        )

        predicted = float(
            np.clip(
                model.predict(input_df)[0],
                0,
                100
            )
        )

        perf = category(predicted)

        conn = get_db()

        try:

            conn.execute(
                """
                INSERT INTO prediction_history
                (
                    student_id,
                    predicted_score,
                    performance_category,
                    model_name,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    round(predicted, 1),
                    perf,
                    "Random Forest Regression",
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            conn.close()

        return render_template(
            "prediction_result.html",
            predicted_score=round(
                predicted,
                1
            ),
            performance=perf,
            student_id=student_id
        )

    except Exception as exc:

        flash(
            f"Prediction error: {exc}",
            "danger"
        )

        return redirect(
            url_for("index") + "#prediction"
        )


# =========================================================
# SINGLE STUDENT
# =========================================================

@app.route(
    "/student/<student_id>"
)
def student(student_id):

    conn = get_db()

    try:

        row = conn.execute(
            """
            SELECT *
            FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()

        history = conn.execute(
            """
            SELECT *
            FROM prediction_history
            WHERE student_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (student_id,)
        ).fetchall()

    finally:

        conn.close()

    if not row:

        flash(
            "Student ID not found.",
            "warning"
        )

        return redirect(
            url_for("index") +
            "#student-analysis"
        )

    return render_template(
        "student.html",
        student=row,
        history=history,
        category=category
    )


# =========================================================
# STUDENTS LIST
# =========================================================

@app.route("/students")
def students():

    q = request.args.get(
        "q",
        ""
    ).strip()

    conn = get_db()

    try:

        if q:

            rows = conn.execute(
                """
                SELECT *
                FROM students
                WHERE student_id ILIKE ?
                   OR student_name ILIKE ?
                   OR department ILIKE ?
                ORDER BY student_id
                """,
                (
                    f"%{q}%",
                    f"%{q}%",
                    f"%{q}%"
                )
            ).fetchall()

        else:

            rows = conn.execute(
                """
                SELECT *
                FROM students
                ORDER BY student_id
                """
            ).fetchall()

    finally:

        conn.close()

    return render_template(
        "students.html",
        students=rows,
        q=q
    )


# =========================================================
# ADD STUDENT
# =========================================================

@app.route(
    "/students/add",
    methods=["GET", "POST"]
)
def add_student():

    if request.method == "GET":

        return render_template(
            "student_form.html",
            mode="add",
            student=None
        )

    data = None

    try:

        data = form_to_student_data(
            request.form
        )

        errors = validate_student(
            data
        )

        if errors:

            for error in errors:

                flash(
                    error,
                    "danger"
                )

            return render_template(
                "student_form.html",
                mode="add",
                student=data
            )

        conn = get_db()

        try:

            placeholders = ",".join(
                ["?"] * len(STUDENT_COLUMNS)
            )

            conn.execute(
                """
                INSERT INTO students
                (
                    """ +
                ",".join(STUDENT_COLUMNS) +
                f"""
                )
                VALUES
                (
                    {placeholders}
                )
                """,
                [
                    data[c]
                    for c in STUDENT_COLUMNS
                ]
            )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            conn.close()

        flash(
            f"Student {data['student_id']} added successfully.",
            "success"
        )

        return redirect(
            url_for("students")
        )

    except psycopg2.IntegrityError:

        flash(
            "Could not add student. "
            "Student ID may already exist.",
            "danger"
        )

        return render_template(
            "student_form.html",
            mode="add",
            student=data
        )

    except Exception as exc:

        flash(
            f"Add student error: {exc}",
            "danger"
        )

        return render_template(
            "student_form.html",
            mode="add",
            student=data or request.form
        )


# =========================================================
# EDIT STUDENT
# =========================================================

@app.route(
    "/students/<student_id>/edit",
    methods=["GET", "POST"]
)
def edit_student(student_id):

    conn = get_db()

    try:

        existing = conn.execute(
            """
            SELECT *
            FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()

    finally:

        conn.close()

    if not existing:

        flash(
            "Student not found.",
            "warning"
        )

        return redirect(
            url_for("students")
        )

    if request.method == "GET":

        return render_template(
            "student_form.html",
            mode="edit",
            student=existing
        )

    data = None

    try:

        data = form_to_student_data(
            request.form
        )

        data["student_id"] = student_id

        errors = validate_student(
            data
        )

        if errors:

            for error in errors:

                flash(
                    error,
                    "danger"
                )

            return render_template(
                "student_form.html",
                mode="edit",
                student=data
            )

        update_cols = [
            c
            for c in STUDENT_COLUMNS
            if c != "student_id"
        ]

        conn = get_db()

        try:

            conn.execute(
                """
                UPDATE students
                SET
                """
                +
                ", ".join(
                    f"{c} = ?"
                    for c in update_cols
                )
                +
                """
                WHERE student_id = ?
                """,
                [
                    data[c]
                    for c in update_cols
                ]
                +
                [student_id]
            )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            conn.close()

        flash(
            f"Student {student_id} updated successfully.",
            "success"
        )

        return redirect(
            url_for("students")
        )

    except Exception as exc:

        flash(
            f"Update error: {exc}",
            "danger"
        )

        return render_template(
            "student_form.html",
            mode="edit",
            student=data or request.form
        )


# =========================================================
# DELETE STUDENT
# =========================================================

@app.post(
    "/students/<student_id>/delete"
)
def delete_student(student_id):

    conn = get_db()

    try:

        cur = conn.execute(
            """
            DELETE FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        )

        conn.commit()

        deleted = cur.rowcount

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    if deleted:

        flash(
            f"Student {student_id} deleted successfully.",
            "success"
        )

    else:

        flash(
            "Student not found.",
            "warning"
        )

    return redirect(
        url_for("students")
    )


# =========================================================
# CSV IMPORT
# =========================================================

@app.route(
    "/students/import",
    methods=["GET", "POST"]
)
def import_csv():

    if request.method == "GET":

        return render_template(
            "import_csv.html"
        )

    uploaded = request.files.get(
        "csv_file"
    )

    if (
        not uploaded
        or not uploaded.filename.lower().endswith(".csv")
    ):

        flash(
            "Please select a CSV file.",
            "danger"
        )

        return redirect(
            url_for("import_csv")
        )

    try:

        df = pd.read_csv(
            uploaded
        )

        missing = [
            c
            for c in STUDENT_COLUMNS
            if c not in df.columns
        ]

        if missing:

            flash(
                "CSV is missing columns: "
                + ", ".join(missing),
                "danger"
            )

            return redirect(
                url_for("import_csv")
            )

        df = df[
            STUDENT_COLUMNS
        ].copy()

        for col in NUMERIC_COLUMNS:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                c
                for c in STUDENT_COLUMNS
                if c != "final_score"
            ]
        )

        df["final_score"] = (
            df["final_score"].fillna(0)
        )

        records = []

        for _, row in df.iterrows():

            record = tuple(
                python_value(row[c])
                for c in STUDENT_COLUMNS
            )

            records.append(record)

        columns = ",".join(
            STUDENT_COLUMNS
        )

        placeholders = ",".join(
            ["%s"] * len(STUDENT_COLUMNS)
        )

        update_columns = [
            c
            for c in STUDENT_COLUMNS
            if c != "student_id"
        ]

        update_sql = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in update_columns
        )

        sql = f"""
            INSERT INTO students
            ({columns})
            VALUES
            ({placeholders})
            ON CONFLICT (student_id)
            DO UPDATE SET
            {update_sql}
        """

        conn = get_db()

        try:

            conn.executemany(
                sql,
                records
            )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            conn.close()

        flash(
            f"Imported {len(records)} student records successfully using Pandas + PostgreSQL.",
            "success"
        )

        return redirect(
            url_for("students")
        )

    except Exception as exc:

        flash(
            f"CSV import error: {exc}",
            "danger"
        )

        return redirect(
            url_for("import_csv")
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/history")
def history():

    conn = get_db()

    try:

        rows = conn.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    finally:

        conn.close()

    return render_template(
        "history.html",
        history=rows
    )


# =========================================================
# VIEW PREDICTION
# =========================================================

@app.route(
    "/history/view/<int:prediction_id>"
)
def view_prediction(prediction_id):

    conn = get_db()

    try:

        prediction = conn.execute(
            """
            SELECT *
            FROM prediction_history
            WHERE id = ?
            """,
            (prediction_id,)
        ).fetchone()

    finally:

        conn.close()

    if not prediction:

        flash(
            "Prediction record not found.",
            "warning"
        )

        return redirect(
            url_for("history")
        )

    return render_template(
        "prediction_detail.html",
        prediction=prediction
    )


# =========================================================
# DELETE PREDICTION
# =========================================================

@app.post(
    "/history/delete/<int:prediction_id>"
)
def delete_prediction(prediction_id):

    conn = get_db()

    try:

        cur = conn.execute(
            """
            DELETE FROM prediction_history
            WHERE id = ?
            """,
            (prediction_id,)
        )

        conn.commit()

        deleted = cur.rowcount

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    if deleted > 0:

        flash(
            "Prediction record deleted successfully.",
            "success"
        )

    else:

        flash(
            "Prediction record not found.",
            "warning"
        )

    return redirect(
        url_for("history")
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    try:

        conn = get_db()

        result = conn.execute(
            "SELECT 1 AS test"
        ).fetchone()

        conn.close()

        if result["test"] == 1:

            return jsonify({
                "status": "ok",
                "database": "Supabase PostgreSQL",
                "model": "Random Forest Regression"
            })

    except Exception as exc:

        return jsonify({
            "status": "error",
            "database": "Supabase PostgreSQL",
            "error": str(exc)
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )