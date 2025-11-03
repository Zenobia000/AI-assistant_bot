#!/usr/bin/env python3
"""
AVATAR - Database Initialization Script
初始化 SQLite 資料庫 schema

Usage:
    poetry run python scripts/init_database.py
    # 或激活環境後: python scripts/init_database.py
"""

import sqlite3
import sys
from pathlib import Path


def create_schema(conn: sqlite3.Connection):
    """Create database schema"""
    cursor = conn.cursor()

    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,

            -- User input
            user_audio_path TEXT NOT NULL,
            user_text TEXT NOT NULL,

            -- AI response
            ai_text TEXT NOT NULL,
            ai_audio_fast_path TEXT,
            ai_audio_hq_path TEXT,

            -- Metadata
            voice_profile_id INTEGER,
            created_at INTEGER NOT NULL,

            UNIQUE(session_id, turn_number)
        )
    """)

    # Create indexes for conversations
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id
        ON conversations(session_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at
        ON conversations(created_at)
    """)

    # Create voice_profiles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            embedding BLOB,
            duration_sec REAL NOT NULL,
            created_at INTEGER NOT NULL,

            UNIQUE(name)
        )
    """)

    # Create index for voice_profiles
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_voice_profile_name
        ON voice_profiles(name)
    """)

    conn.commit()
    print("[OK] Database schema created successfully")


def enable_wal_mode(conn: sqlite3.Connection):
    """Enable WAL mode for better concurrency"""
    cursor = conn.cursor()

    # Enable WAL mode
    cursor.execute("PRAGMA journal_mode=WAL")
    result = cursor.fetchone()
    print(f"[INFO] Journal mode: {result[0]}")

    # Set busy timeout to 5 seconds
    cursor.execute("PRAGMA busy_timeout=5000")

    conn.commit()
    print("[OK] WAL mode enabled")


def verify_schema(conn: sqlite3.Connection):
    """Verify schema is correctly created"""
    cursor = conn.cursor()

    # Check tables
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)

    tables = cursor.fetchall()
    print(f"\n[INFO] Tables created: {len(tables)}")
    for table in tables:
        print(f"       - {table[0]}")

        # Count columns
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        print(f"         Columns: {len(columns)}")

    # Check indexes
    cursor.execute("""
        SELECT name, tbl_name FROM sqlite_master
        WHERE type='index' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)

    indexes = cursor.fetchall()
    print(f"\n[INFO] Indexes created: {len(indexes)}")
    for index in indexes:
        print(f"       - {index[0]} on {index[1]}")


def main():
    """Main initialization function"""
    print("[CHECK] AVATAR Database Initialization")

    # Database file path
    db_path = Path("app.db")

    # Check if database already exists
    if db_path.exists():
        print(f"[WARN] Database file already exists: {db_path}")
        response = input("       Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("[INFO] Initialization cancelled")
            return 0

        # Backup existing database
        backup_path = Path(f"app.db.backup")
        db_path.rename(backup_path)
        print(f"[INFO] Existing database backed up to: {backup_path}")

    # Create database connection
    print(f"\n[INFO] Creating database: {db_path}")
    conn = sqlite3.connect(str(db_path))

    try:
        # Create schema
        create_schema(conn)

        # Enable WAL mode
        enable_wal_mode(conn)

        # Verify schema
        verify_schema(conn)

        print("\n[OK] Database initialization complete!")
        print(f"     Database file: {db_path.absolute()}")
        print(f"     Size: {db_path.stat().st_size} bytes")

        return 0

    except Exception as e:
        print(f"\n[FAIL] Database initialization failed: {e}")
        return 1

    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
