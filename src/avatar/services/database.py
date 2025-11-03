"""
Database operations for conversations and voice profiles

Provides async interface to SQLite database using aiosqlite.
"""

import time
from pathlib import Path
from typing import Optional

import aiosqlite
import structlog

from avatar.core.config import config

logger = structlog.get_logger()


class DatabaseService:
    """
    Async database service for AVATAR

    Handles all database operations for conversations and voice profiles.
    """

    def __init__(self, db_path: Path = config.DATABASE_PATH):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Establish database connection"""
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            logger.info("db.connected", path=str(self.db_path))

    async def close(self):
        """Close database connection"""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("db.closed")

    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()

    # Conversation operations

    async def save_conversation(
        self,
        session_id: str,
        turn_number: int,
        user_audio_path: str,
        user_text: str,
        ai_text: str,
        ai_audio_fast_path: Optional[str] = None,
        ai_audio_hq_path: Optional[str] = None,
        voice_profile_id: Optional[int] = None,
    ) -> int:
        """
        Save a conversation turn to database

        Returns: conversation ID
        """
        if not self._conn:
            await self.connect()

        created_at = int(time.time())

        cursor = await self._conn.execute(
            """
            INSERT INTO conversations (
                session_id, turn_number,
                user_audio_path, user_text,
                ai_text, ai_audio_fast_path, ai_audio_hq_path,
                voice_profile_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_number,
                user_audio_path,
                user_text,
                ai_text,
                ai_audio_fast_path,
                ai_audio_hq_path,
                voice_profile_id,
                created_at,
            ),
        )

        await self._conn.commit()
        conversation_id = cursor.lastrowid

        logger.info(
            "db.conversation.saved",
            id=conversation_id,
            session_id=session_id,
            turn=turn_number,
        )

        return conversation_id

    async def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> list[dict]:
        """
        Get conversation history for a session

        Returns: List of conversation turns (newest first)
        """
        if not self._conn:
            await self.connect()

        cursor = await self._conn.execute(
            """
            SELECT
                id, session_id, turn_number,
                user_text, ai_text,
                ai_audio_fast_path, ai_audio_hq_path,
                voice_profile_id, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY turn_number DESC
            LIMIT ?
            """,
            (session_id, limit),
        )

        rows = await cursor.fetchall()
        conversations = [dict(row) for row in rows]

        logger.debug(
            "db.conversation.fetched",
            session_id=session_id,
            count=len(conversations),
        )

        return conversations

    async def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """
        Get recent conversation sessions

        Returns: List of sessions with metadata
        """
        if not self._conn:
            await self.connect()

        cursor = await self._conn.execute(
            """
            SELECT
                session_id,
                COUNT(*) as turn_count,
                MIN(created_at) as started_at,
                MAX(created_at) as last_updated
            FROM conversations
            GROUP BY session_id
            ORDER BY last_updated DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = await cursor.fetchall()
        sessions = [dict(row) for row in rows]

        logger.debug("db.sessions.fetched", count=len(sessions))

        return sessions

    # Voice profile operations

    async def create_voice_profile(
        self,
        name: str,
        audio_path: str,
        duration_sec: float,
        embedding: Optional[bytes] = None,
    ) -> int:
        """
        Create a new voice profile

        Returns: voice profile ID
        """
        if not self._conn:
            await self.connect()

        created_at = int(time.time())

        cursor = await self._conn.execute(
            """
            INSERT INTO voice_profiles (
                name, audio_path, embedding, duration_sec, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (name, audio_path, embedding, duration_sec, created_at),
        )

        await self._conn.commit()
        profile_id = cursor.lastrowid

        logger.info(
            "db.voice_profile.created",
            id=profile_id,
            name=name,
            duration=duration_sec,
        )

        return profile_id

    async def get_voice_profile(self, profile_id: int) -> Optional[dict]:
        """Get voice profile by ID"""
        if not self._conn:
            await self.connect()

        cursor = await self._conn.execute(
            """
            SELECT id, name, audio_path, embedding, duration_sec, created_at
            FROM voice_profiles
            WHERE id = ?
            """,
            (profile_id,),
        )

        row = await cursor.fetchone()

        if row:
            logger.debug("db.voice_profile.fetched", id=profile_id)
            return dict(row)
        else:
            logger.warning("db.voice_profile.not_found", id=profile_id)
            return None

    async def get_voice_profile_by_name(self, name: str) -> Optional[dict]:
        """Get voice profile by name"""
        if not self._conn:
            await self.connect()

        cursor = await self._conn.execute(
            """
            SELECT id, name, audio_path, embedding, duration_sec, created_at
            FROM voice_profiles
            WHERE name = ?
            """,
            (name,),
        )

        row = await cursor.fetchone()

        if row:
            logger.debug("db.voice_profile.fetched_by_name", name=name)
            return dict(row)
        else:
            return None

    async def list_voice_profiles(self) -> list[dict]:
        """List all voice profiles"""
        if not self._conn:
            await self.connect()

        cursor = await self._conn.execute(
            """
            SELECT id, name, audio_path, duration_sec, created_at
            FROM voice_profiles
            ORDER BY created_at DESC
            """
        )

        rows = await cursor.fetchall()
        profiles = [dict(row) for row in rows]

        logger.debug("db.voice_profiles.listed", count=len(profiles))

        return profiles

    async def delete_voice_profile(self, profile_id: int) -> bool:
        """
        Delete a voice profile

        Returns: True if deleted, False if not found
        """
        if not self._conn:
            await self.connect()

        cursor = await self._conn.execute(
            """
            DELETE FROM voice_profiles WHERE id = ?
            """,
            (profile_id,),
        )

        await self._conn.commit()
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info("db.voice_profile.deleted", id=profile_id)
        else:
            logger.warning("db.voice_profile.delete_failed", id=profile_id)

        return deleted

    # New voice profile methods for UUID-based API
    async def create_voice_profile_v2(self, profile_data: dict) -> str:
        """
        Create voice profile with UUID (API v2)

        Args:
            profile_data: Profile data with fields:
                - id: UUID string
                - name: Profile name
                - description: Optional description
                - reference_text: Optional reference text
                - audio_path: Path to audio file
                - file_size: File size in bytes
                - created_at: Creation timestamp
                - updated_at: Update timestamp

        Returns:
            Profile UUID
        """
        if not self._conn:
            await self.connect()

        # Check if table needs migration
        await self._ensure_voice_profiles_v2_schema()

        created_at = profile_data['created_at'].timestamp()
        updated_at = profile_data['updated_at'].timestamp()

        await self._conn.execute(
            """
            INSERT INTO voice_profiles_v2 (
                id, name, description, reference_text, audio_path,
                file_size, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_data['id'],
                profile_data['name'],
                profile_data.get('description'),
                profile_data.get('reference_text'),
                profile_data['audio_path'],
                profile_data['file_size'],
                created_at,
                updated_at
            ),
        )

        await self._conn.commit()

        logger.info(
            "db.voice_profile_v2.created",
            id=profile_data['id'],
            name=profile_data['name'],
            file_size=profile_data['file_size']
        )

        return profile_data['id']

    async def get_voice_profile_v2(self, profile_id: str) -> Optional[dict]:
        """Get voice profile by UUID"""
        if not self._conn:
            await self.connect()

        await self._ensure_voice_profiles_v2_schema()

        cursor = await self._conn.execute(
            """
            SELECT id, name, description, reference_text, audio_path,
                   file_size, created_at, updated_at
            FROM voice_profiles_v2
            WHERE id = ?
            """,
            (profile_id,),
        )

        row = await cursor.fetchone()
        if row:
            from datetime import datetime
            return {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'reference_text': row[3],
                'audio_path': row[4],
                'file_size': row[5],
                'created_at': datetime.fromtimestamp(row[6]),
                'updated_at': datetime.fromtimestamp(row[7])
            }
        return None

    async def get_voice_profiles_v2(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get voice profiles with pagination"""
        if not self._conn:
            await self.connect()

        await self._ensure_voice_profiles_v2_schema()

        cursor = await self._conn.execute(
            """
            SELECT id, name, description, reference_text, audio_path,
                   file_size, created_at, updated_at
            FROM voice_profiles_v2
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        rows = await cursor.fetchall()
        from datetime import datetime

        profiles = []
        for row in rows:
            profiles.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'reference_text': row[3],
                'audio_path': row[4],
                'file_size': row[5],
                'created_at': datetime.fromtimestamp(row[6]),
                'updated_at': datetime.fromtimestamp(row[7])
            })

        logger.debug("db.voice_profiles_v2.listed", count=len(profiles))
        return profiles

    async def count_voice_profiles_v2(self) -> int:
        """Count total voice profiles"""
        if not self._conn:
            await self.connect()

        await self._ensure_voice_profiles_v2_schema()

        cursor = await self._conn.execute("SELECT COUNT(*) FROM voice_profiles_v2")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def update_voice_profile_v2(self, profile_id: str, update_data: dict) -> bool:
        """Update voice profile"""
        if not self._conn:
            await self.connect()

        await self._ensure_voice_profiles_v2_schema()

        # Build dynamic update query
        fields = []
        values = []

        for field in ['name', 'description', 'reference_text', 'audio_path', 'file_size']:
            if field in update_data:
                fields.append(f"{field} = ?")
                values.append(update_data[field])

        if 'updated_at' in update_data:
            fields.append("updated_at = ?")
            values.append(update_data['updated_at'].timestamp())

        if not fields:
            return True  # Nothing to update

        values.append(profile_id)

        cursor = await self._conn.execute(
            f"UPDATE voice_profiles_v2 SET {', '.join(fields)} WHERE id = ?",
            values
        )

        await self._conn.commit()

        success = cursor.rowcount > 0
        if success:
            logger.info("db.voice_profile_v2.updated", id=profile_id)
        else:
            logger.warning("db.voice_profile_v2.update_failed", id=profile_id)

        return success

    async def delete_voice_profile_v2(self, profile_id: str) -> bool:
        """Delete voice profile by UUID"""
        if not self._conn:
            await self.connect()

        await self._ensure_voice_profiles_v2_schema()

        cursor = await self._conn.execute(
            "DELETE FROM voice_profiles_v2 WHERE id = ?",
            (profile_id,)
        )

        await self._conn.commit()

        success = cursor.rowcount > 0
        if success:
            logger.info("db.voice_profile_v2.deleted", id=profile_id)
        else:
            logger.warning("db.voice_profile_v2.delete_failed", id=profile_id)

        return success

    async def _ensure_voice_profiles_v2_schema(self):
        """Ensure voice_profiles_v2 table exists"""
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_profiles_v2 (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                reference_text TEXT,
                audio_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,

                UNIQUE(name)
            )
            """
        )
        await self._conn.commit()


# Global database service instance
db = DatabaseService()


async def get_database_service() -> DatabaseService:
    """
    Dependency injection for database service

    Returns:
        Database service instance
    """
    if not db._conn:
        await db.connect()
    return db
