
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "student_performance.db"
CSV = ROOT / "data" / "student_performance_500.csv"

df = pd.read_csv(CSV)
conn = sqlite3.connect(DB)
df.to_sql("students", conn, if_exists="replace", index=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    predicted_score REAL,
    performance_category TEXT,
    model_name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()
print("SQLite database initialized:", DB)
