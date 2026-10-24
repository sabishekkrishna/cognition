import sqlite3
import json
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

DATABASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
DATABASE_PATH = os.path.join(DATABASE_DIR, "monitor.db")

def init_db():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """)

    # Baselines table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS baselines (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL, -- 'image' or 'audio'
        description TEXT,
        is_active INTEGER DEFAULT 0,
        sample_count INTEGER DEFAULT 0,
        stats_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """)

    # Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        subsystem TEXT NOT NULL, -- 'image', 'audio', 'llm', 'system'
        severity TEXT NOT NULL,  -- 'normal', 'caution', 'warning', 'critical'
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        metric_name TEXT,
        measured_value REAL,
        expected_range TEXT,
        recommendation TEXT,
        metadata_json TEXT
    )
    """)

    # LLM Responses & Feedback table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS llm_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL,
        input_subsystem TEXT,
        input_risk_score REAL,
        response_risk_score REAL,
        warnings_json TEXT,
        user_rating TEXT, -- 'correct', 'incorrect', 'uncertain', null
        user_notes TEXT
    )
    """)

    # Datasets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        description TEXT,
        classes_json TEXT NOT NULL,
        sample_count INTEGER NOT NULL,
        train_count INTEGER NOT NULL,
        val_count INTEGER NOT NULL,
        test_count INTEGER NOT NULL,
        created_at REAL NOT NULL
    )
    """)

    # Model versions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        is_active INTEGER DEFAULT 0,
        checkpoint_path TEXT,
        metrics_json TEXT,
        created_at REAL NOT NULL
    )
    """)

    conn.commit()

    # Insert default settings if not exists
    default_settings = {
        "llm_provider": "mock",
        "ollama_url": "http://127.0.0.1:11434",
        "openai_compat_url": "http://127.0.0.1:1234/v1",
        "llama_cpp_url": "http://127.0.0.1:8080/v1",
        "custom_llm_url": "http://127.0.0.1:3000",
        "custom_llm_path": "",
        "custom_llm_type": "auto",
        "custom_llm_model": "custom-model",
        "active_model": "mock-llama-3b",
        "camera_sample_fps": "2",
        "sharpness_min_threshold": "60.0",
        "snr_min_threshold": "12.0",
        "noise_max_threshold": "25.0",
        "clipping_max_threshold": "2.0",
        "hysteresis_window_size": "5",
        "hysteresis_threshold": "3",
        "data_retention_days": "30"
    }

    # Fix legacy typo in ollama_url if present
    cursor.execute("UPDATE settings SET value = 'http://127.0.0.1:11434' WHERE key = 'ollama_url' AND value = 'http://127.0.0.1:1434'")

    for k, v in default_settings.items():
        cursor.execute("SELECT key FROM settings WHERE key = ?", (k,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                           (k, v, time.time()))

    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Helper functions for Settings
def get_all_settings() -> Dict[str, str]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings")
    rows = cursor.fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}

def set_setting(key: str, value: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                   (key, value, time.time()))
    conn.commit()
    conn.close()

# Helper functions for Events
def add_event(subsystem: str, severity: str, title: str, message: str,
              metric_name: Optional[str] = None, measured_value: Optional[float] = None,
              expected_range: Optional[str] = None, recommendation: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO events (timestamp, subsystem, severity, title, message, metric_name, measured_value, expected_range, recommendation, metadata_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        time.time(),
        subsystem,
        severity,
        title,
        message,
        metric_name,
        measured_value,
        expected_range,
        recommendation,
        json.dumps(metadata or {})
    ))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id

def get_recent_events(limit: int = 50, subsystem: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if subsystem and subsystem != "all":
        query += " AND subsystem = ?"
        params.append(subsystem)
    if severity and severity != "all":
        query += " AND severity = ?"
        params.append(severity)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    events = []
    for r in rows:
        events.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "time_formatted": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "subsystem": r["subsystem"],
            "severity": r["severity"],
            "title": r["title"],
            "message": r["message"],
            "metric_name": r["metric_name"],
            "measured_value": r["measured_value"],
            "expected_range": r["expected_range"],
            "recommendation": r["recommendation"],
            "metadata": json.loads(r["metadata_json"] or "{}")
        })
    return events

def clear_events():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events")
    conn.commit()
    conn.close()

# Helper functions for Baselines
def get_baselines(baseline_type: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    if baseline_type:
        cursor.execute("SELECT * FROM baselines WHERE type = ? ORDER BY created_at DESC", (baseline_type,))
    else:
        cursor.execute("SELECT * FROM baselines ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "name": r["name"],
            "type": r["type"],
            "description": r["description"],
            "is_active": bool(r["is_active"]),
            "sample_count": r["sample_count"],
            "stats": json.loads(r["stats_json"] or "{}"),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"]
        })
    return result

def get_active_baseline(baseline_type: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM baselines WHERE type = ? AND is_active = 1 LIMIT 1", (baseline_type,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "type": row["type"],
        "description": row["description"],
        "is_active": bool(row["is_active"]),
        "sample_count": row["sample_count"],
        "stats": json.loads(row["stats_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }

def save_baseline(id: str, name: str, baseline_type: str, description: str,
                  sample_count: int, stats: Dict[str, Any], set_active: bool = True) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()

    if set_active:
        cursor.execute("UPDATE baselines SET is_active = 0 WHERE type = ?", (baseline_type,))

    cursor.execute("""
    INSERT OR REPLACE INTO baselines (id, name, type, description, is_active, sample_count, stats_json, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        id,
        name,
        baseline_type,
        description,
        1 if set_active else 0,
        sample_count,
        json.dumps(stats),
        now,
        now
    ))
    conn.commit()
    conn.close()
    return True

def set_active_baseline(baseline_id: str) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT type FROM baselines WHERE id = ?", (baseline_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
    b_type = row["type"]
    cursor.execute("UPDATE baselines SET is_active = 0 WHERE type = ?", (b_type,))
    cursor.execute("UPDATE baselines SET is_active = 1 WHERE id = ?", (baseline_id,))
    conn.commit()
    conn.close()
    return True

def delete_baseline(baseline_id: str) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM baselines WHERE id = ?", (baseline_id,))
    conn.commit()
    conn.close()
    return True

# Initialize database on module load
init_db()
