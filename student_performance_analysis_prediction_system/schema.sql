-- Student Performance Analysis & Prediction System
-- SQLite database schema.
-- No MySQL and no login/authentication are used.

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    student_name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    department TEXT NOT NULL,
    semester INTEGER NOT NULL,
    attendance_percentage REAL NOT NULL,
    study_hours_per_day REAL NOT NULL,
    previous_exam_score REAL NOT NULL,
    midterm_score REAL NOT NULL,
    assignment_percentage REAL NOT NULL,
    practical_lab_score REAL NOT NULL,
    sleep_hours_per_day REAL NOT NULL,
    online_learning_hours REAL NOT NULL,
    previous_backlogs INTEGER NOT NULL,
    internet_access TEXT NOT NULL,
    quiz_score REAL NOT NULL,
    class_participation_score REAL NOT NULL,
    final_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    predicted_score REAL NOT NULL,
    performance_category TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- CRUD examples used by the Flask application:
-- CREATE: INSERT INTO students (...) VALUES (...);
-- READ:   SELECT * FROM students WHERE student_id = ?;
-- UPDATE: UPDATE students SET attendance_percentage = ? WHERE student_id = ?;
-- DELETE: DELETE FROM students WHERE student_id = ?;
-- CSV: Pandas reads CSV, then SQLite executemany/INSERT OR REPLACE stores rows.
