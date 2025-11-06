"""
Conversation History API

REST endpoints for managing and retrieving conversation history.
Provides access to chat sessions and conversation turns for analysis and review.
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from avatar.core.config import config
from avatar.core.security import verify_api_token, optional_api_token, safe_error_response
from avatar.services.database import get_database_service
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = structlog.get_logger()

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# Models
class ConversationTurn(BaseModel):
    """Individual conversation turn model"""
    id: int
    session_id: str
    turn_number: int
    user_text: str
    ai_text: str
    user_audio_path: Optional[str]
    ai_audio_fast_path: Optional[str]
    ai_audio_hq_path: Optional[str]
    voice_profile_id: Optional[int]
    created_at: datetime
    processing_time_ms: Optional[float]


class ConversationSession(BaseModel):
    """Conversation session summary model"""
    session_id: str
    first_message: str
    turn_count: int
    created_at: datetime
    last_activity: datetime
    voice_profile_name: Optional[str]


class ConversationHistory(BaseModel):
    """Conversation history response"""
    session_id: str
    turns: List[ConversationTurn]
    total_turns: int
    session_created: datetime


class ConversationList(BaseModel):
    """List of conversation sessions"""
    sessions: List[ConversationSession]
    total: int
    page: int
    per_page: int


# API Endpoints
@router.get("/sessions", response_model=ConversationList)
@limiter.limit("20/minute")
async def list_conversation_sessions(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    authenticated: bool = Depends(optional_api_token),
    db = Depends(get_database_service)
):
    """
    List conversation sessions with pagination

    Args:
        page: Page number (1-based)
        per_page: Items per page (max 100)
        authenticated: Authentication status (optional)
        db: Database service

    Returns:
        List of conversation sessions
    """
    logger.info("conversations.list_sessions", page=page, per_page=per_page)

    try:
        # Calculate offset
        offset = (page - 1) * per_page

        # Get conversation sessions
        sessions = await db.get_recent_conversation_sessions(
            limit=per_page,
            offset=offset
        )

        # Get total count (for pagination info)
        total_sessions = await db.count_conversation_sessions()

        # Convert to response models
        session_responses = []
        for session in sessions:
            session_responses.append(ConversationSession(
                session_id=session['session_id'],
                first_message=session['first_user_message'][:100],  # Truncate for preview
                turn_count=session['turn_count'],
                created_at=datetime.fromtimestamp(session['first_created_at']),
                last_activity=datetime.fromtimestamp(session['last_created_at']),
                voice_profile_name=session.get('voice_profile_name')
            ))

        logger.info(
            "conversations.sessions_listed",
            returned=len(sessions),
            total=total_sessions,
            page=page
        )

        return ConversationList(
            sessions=session_responses,
            total=total_sessions,
            page=page,
            per_page=per_page
        )

    except Exception as e:
        logger.error("conversations.list_sessions_failed", error=str(e))
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("/{session_id}", response_model=ConversationHistory)
@limiter.limit("30/minute")
async def get_conversation_history(
    request: Request,
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max conversation turns"),
    authenticated: bool = Depends(optional_api_token),
    db = Depends(get_database_service)
):
    """
    Get conversation history for a specific session

    Args:
        session_id: Session identifier
        limit: Maximum number of turns to return
        authenticated: Authentication status (optional)
        db: Database service

    Returns:
        Complete conversation history
    """
    logger.info("conversations.get_history", session_id=session_id, limit=limit)

    try:
        # Get conversation turns
        turns = await db.get_conversation_history(session_id, limit=limit)

        if not turns:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation found for session {session_id}"
            )

        # Convert to response models
        turn_responses = []
        for turn in turns:
            turn_responses.append(ConversationTurn(
                id=turn['id'],
                session_id=turn['session_id'],
                turn_number=turn['turn_number'],
                user_text=turn['user_text'],
                ai_text=turn['ai_text'],
                user_audio_path=turn.get('user_audio_path'),
                ai_audio_fast_path=turn.get('ai_audio_fast_path'),
                ai_audio_hq_path=turn.get('ai_audio_hq_path'),
                voice_profile_id=turn.get('voice_profile_id'),
                created_at=datetime.fromtimestamp(turn['created_at']),
                processing_time_ms=turn.get('processing_time_ms')
            ))

        session_created = min(turn.created_at for turn in turn_responses)

        logger.info(
            "conversations.history_retrieved",
            session_id=session_id,
            turns=len(turn_responses)
        )

        return ConversationHistory(
            session_id=session_id,
            turns=turn_responses,
            total_turns=len(turn_responses),
            session_created=session_created
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversations.get_history_failed", error=str(e), session_id=session_id)
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("/{session_id}/audio/{turn_number}")
@limiter.limit("10/minute")
async def get_conversation_audio(
    request: Request,
    session_id: str,
    turn_number: int,
    audio_type: str = Query("ai_fast", regex="^(user|ai_fast|ai_hq)$"),
    authenticated: bool = Depends(optional_api_token),
    db = Depends(get_database_service)
):
    """
    Download audio file from specific conversation turn

    Args:
        session_id: Session identifier
        turn_number: Turn number within session
        audio_type: Type of audio (user, ai_fast, ai_hq)
        authenticated: Authentication status (optional)
        db: Database service

    Returns:
        Audio file response
    """
    logger.info(
        "conversations.get_audio",
        session_id=session_id,
        turn_number=turn_number,
        audio_type=audio_type
    )

    try:
        # Get specific conversation turn
        turns = await db.get_conversation_history(session_id, limit=1000)  # Get all to find specific turn
        turn = next((t for t in turns if t['turn_number'] == turn_number), None)

        if not turn:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation turn {turn_number} not found in session {session_id}"
            )

        # Get audio path based on type
        audio_path_key = f"{audio_type.replace('_', '_')}_audio_path"
        if audio_type == "user":
            audio_path = turn.get("user_audio_path")
        elif audio_type == "ai_fast":
            audio_path = turn.get("ai_audio_fast_path")
        elif audio_type == "ai_hq":
            audio_path = turn.get("ai_audio_hq_path")
        else:
            raise HTTPException(status_code=400, detail="Invalid audio type")

        if not audio_path:
            raise HTTPException(
                status_code=404,
                detail=f"No {audio_type} audio available for this turn"
            )

        # Check file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Audio file not found on disk"
            )

        logger.info(
            "conversations.audio_served",
            session_id=session_id,
            turn_number=turn_number,
            audio_type=audio_type,
            file_size=audio_file.stat().st_size
        )

        return FileResponse(
            path=str(audio_file),
            media_type="audio/wav",
            filename=f"{session_id}_{turn_number}_{audio_type}.wav"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversations.get_audio_failed", error=str(e), session_id=session_id)
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)


@router.delete("/{session_id}")
@limiter.limit("5/minute")
async def delete_conversation_session(
    request: Request,
    session_id: str,
    authenticated: bool = Depends(verify_api_token),  # Deletion requires auth
    db = Depends(get_database_service)
):
    """
    Delete a conversation session and all its turns

    Args:
        session_id: Session identifier to delete
        authenticated: Authentication status (required)
        db: Database service

    Returns:
        Success message
    """
    logger.info("conversations.delete_session", session_id=session_id)

    try:
        # Get conversation history to find audio files for cleanup
        turns = await db.get_conversation_history(session_id, limit=1000)

        if not turns:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation session {session_id} not found"
            )

        # Delete audio files
        deleted_files = 0
        for turn in turns:
            for audio_path_key in ['user_audio_path', 'ai_audio_fast_path', 'ai_audio_hq_path']:
                audio_path = turn.get(audio_path_key)
                if audio_path:
                    try:
                        Path(audio_path).unlink(missing_ok=True)
                        deleted_files += 1
                    except Exception:
                        # Silent failure for file cleanup
                        pass

        # Delete from database
        success = await db.delete_conversation_session(session_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Session not found or already deleted"
            )

        logger.info(
            "conversations.session_deleted",
            session_id=session_id,
            turns_deleted=len(turns),
            files_deleted=deleted_files
        )

        return {
            "message": f"Conversation session {session_id} deleted successfully",
            "turns_deleted": len(turns),
            "files_deleted": deleted_files
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversations.delete_session_failed", error=str(e), session_id=session_id)
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("/sessions/search", response_model=ConversationList)
@limiter.limit("15/minute")
async def search_conversations(
    request: Request,
    query: str = Query(..., min_length=1, max_length=100, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=50, description="Items per page"),
    authenticated: bool = Depends(optional_api_token),
    db = Depends(get_database_service)
):
    """
    Search conversations by text content

    Args:
        query: Search query string
        page: Page number (1-based)
        per_page: Items per page (max 50 for search)
        authenticated: Authentication status (optional)
        db: Database service

    Returns:
        Matching conversation sessions
    """
    logger.info("conversations.search", query=query[:50], page=page)

    try:
        # Input validation
        if len(query.strip()) < 1:
            raise HTTPException(
                status_code=400,
                detail="Search query cannot be empty"
            )

        # Calculate offset
        offset = (page - 1) * per_page

        # Search conversations
        sessions = await db.search_conversations(
            query=query.strip(),
            limit=per_page,
            offset=offset
        )

        total_found = await db.count_search_results(query.strip())

        # Convert to response format
        session_responses = []
        for session in sessions:
            session_responses.append(ConversationSession(
                session_id=session['session_id'],
                first_message=session['matched_text'][:100],  # Show matched content
                turn_count=session['turn_count'],
                created_at=datetime.fromtimestamp(session['created_at']),
                last_activity=datetime.fromtimestamp(session['last_activity']),
                voice_profile_name=session.get('voice_profile_name')
            ))

        logger.info(
            "conversations.search_complete",
            query=query[:50],
            found=len(sessions),
            total=total_found
        )

        return ConversationList(
            sessions=session_responses,
            total=total_found,
            page=page,
            per_page=per_page
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversations.search_failed", error=str(e), query=query[:50])
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("/sessions/stats")
@limiter.limit("10/minute")
async def get_conversation_stats(
    request: Request,
    authenticated: bool = Depends(optional_api_token),
    db = Depends(get_database_service)
):
    """
    Get conversation statistics and metrics

    Args:
        authenticated: Authentication status (optional)
        db: Database service

    Returns:
        Conversation statistics
    """
    logger.info("conversations.get_stats")

    try:
        # Get basic stats
        total_sessions = await db.count_conversation_sessions()
        total_turns = await db.count_conversation_turns()

        # Calculate average turns per session
        avg_turns = (total_turns / total_sessions) if total_sessions > 0 else 0

        # Get recent activity (last 24 hours)
        recent_sessions = await db.count_recent_sessions(hours=24)
        recent_turns = await db.count_recent_turns(hours=24)

        stats = {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "average_turns_per_session": round(avg_turns, 1),
            "recent_24h": {
                "sessions": recent_sessions,
                "turns": recent_turns
            },
            "generated_at": datetime.utcnow().isoformat()
        }

        logger.info("conversations.stats_generated", **stats["recent_24h"])

        return stats

    except Exception as e:
        logger.error("conversations.get_stats_failed", error=str(e))
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)


@router.post("/{session_id}/export")
@limiter.limit("3/minute")  # Limited export for resource protection
async def export_conversation_session(
    request: Request,
    session_id: str,
    format: str = Query("json", regex="^(json|txt)$", description="Export format"),
    authenticated: bool = Depends(verify_api_token),  # Export requires auth
    db = Depends(get_database_service)
):
    """
    Export conversation session in specified format

    Args:
        session_id: Session to export
        format: Export format (json or txt)
        authenticated: Authentication status (required)
        db: Database service

    Returns:
        Exported conversation file
    """
    logger.info("conversations.export", session_id=session_id, format=format)

    try:
        # Get conversation history
        turns = await db.get_conversation_history(session_id, limit=1000)

        if not turns:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation session {session_id} not found"
            )

        # Generate export content
        if format == "json":
            import json
            export_data = {
                "session_id": session_id,
                "exported_at": datetime.utcnow().isoformat(),
                "turns": [
                    {
                        "turn": turn['turn_number'],
                        "timestamp": datetime.fromtimestamp(turn['created_at']).isoformat(),
                        "user": turn['user_text'],
                        "ai": turn['ai_text'],
                        "processing_time_ms": turn.get('processing_time_ms')
                    }
                    for turn in turns
                ]
            }
            content = json.dumps(export_data, indent=2, ensure_ascii=False)
            media_type = "application/json"
            filename = f"conversation_{session_id}.json"

        else:  # txt format
            content_lines = [
                f"Conversation Export: {session_id}",
                f"Exported: {datetime.utcnow().isoformat()}",
                "=" * 50,
                ""
            ]

            for turn in turns:
                timestamp = datetime.fromtimestamp(turn['created_at'])
                content_lines.extend([
                    f"Turn {turn['turn_number']} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"User: {turn['user_text']}",
                    f"AI: {turn['ai_text']}",
                    ""
                ])

            content = "\n".join(content_lines)
            media_type = "text/plain"
            filename = f"conversation_{session_id}.txt"

        # Save temporary file for download
        export_dir = config.AUDIO_DIR / "exports"
        export_dir.mkdir(exist_ok=True)

        export_path = export_dir / filename
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            "conversations.export_complete",
            session_id=session_id,
            format=format,
            turns_exported=len(turns),
            file_size=export_path.stat().st_size
        )

        return FileResponse(
            path=str(export_path),
            media_type=media_type,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversations.export_failed", error=str(e), session_id=session_id)
        safe_detail = safe_error_response(str(e))
        raise HTTPException(status_code=500, detail=safe_detail)