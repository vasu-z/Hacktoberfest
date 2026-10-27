import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from backend.config import settings


class MemoryCreateRequest(BaseModel):
    user_id: str
    name: str
    description: str
    location_label: Optional[str] = None


class MemoryUpdateRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    location_label: Optional[str] = None


class MemoryRecord(BaseModel):
    id: str
    user_id: str
    name: str
    description: str
    location_label: Optional[str] = None
    created_at: str
    updated_at: str


class SpatialMemoryService:
    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else Path(settings.saarthi_database_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spatial_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    location_label TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_spatial_memories_user_id
                ON spatial_memories(user_id);
                """
            )
            conn.commit()

    def create_memory(
        self,
        user_id: str,
        name: str,
        description: str,
        location_label: Optional[str] = None,
    ) -> MemoryRecord:
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO spatial_memories (id, user_id, name, description, location_label, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, user_id, name, description, location_label, now, now),
            )
            conn.commit()
        return MemoryRecord(
            id=record_id,
            user_id=user_id,
            name=name,
            description=description,
            location_label=location_label,
            created_at=now,
            updated_at=now,
        )

    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, user_id, name, description, location_label, created_at, updated_at
                FROM spatial_memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
        return [
            MemoryRecord(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                description=row["description"],
                location_label=row["location_label"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_memory(self, memory_id: str, user_id: str) -> Optional[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, user_id, name, description, location_label, created_at, updated_at
                FROM spatial_memories
                WHERE id = ? AND user_id = ?
                """,
                (memory_id, user_id),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            location_label=row["location_label"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        location_label: Optional[str] = None,
        clear_location_label: bool = False,
    ) -> Optional[MemoryRecord]:
        existing = self.get_memory(memory_id, user_id)
        if not existing:
            return None

        updated_name = name if name is not None else existing.name
        updated_description = description if description is not None else existing.description
        if clear_location_label:
            updated_location = None
        elif location_label is not None:
            updated_location = location_label
        else:
            updated_location = existing.location_label

        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE spatial_memories
                SET name = ?, description = ?, location_label = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (updated_name, updated_description, updated_location, now, memory_id, user_id),
            )
            conn.commit()

        return MemoryRecord(
            id=memory_id,
            user_id=user_id,
            name=updated_name,
            description=updated_description,
            location_label=updated_location,
            created_at=existing.created_at,
            updated_at=now,
        )

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM spatial_memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0
