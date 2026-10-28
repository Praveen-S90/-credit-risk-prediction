import sqlite3
from pathlib import Path

# Store the database beside this Python file.
DATABASE_PATH = Path(__file__).resolve().parent / "credit_risk.db"


def get_connection():
    """Open a SQLite connection and return rows as dictionary-like objects."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the prediction history table if it does not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checking_status TEXT,
                duration INTEGER,
                credit_amount REAL,
                age INTEGER,
                risk_score REAL,
                risk_level TEXT,
                high_risk_probability REAL,
                low_risk_probability REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_prediction(applicant, result):
    """
    Save one prediction.

    `applicant` and `result` must be dictionaries. The field names should
    match the data returned by your existing app.py prediction route.
    """
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                checking_status,
                duration,
                credit_amount,
                age,
                risk_score,
                risk_level,
                high_risk_probability,
                low_risk_probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                applicant.get("checking_status"),
                applicant.get("duration"),
                applicant.get("credit_amount"),
                applicant.get("age"),
                result.get("risk_score"),
                result.get("risk_level"),
                result.get("high_risk_probability"),
                result.get("low_risk_probability"),
            ),
        )


def get_prediction_history(limit=100):
    """Return the most recent predictions, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized at: {DATABASE_PATH}")
