"""
Voice Profile Management API

REST endpoints for managing voice profiles used in TTS synthesis.
Handles upload, storage, validation, and management of voice samples.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from avatar.core.config import config
from avatar.core.security import verify_api_token, validate_input_string, safe_error_response
from avatar.services.database import get_database_service

logger = structlog.get_logger()

router = APIRouter(prefix="/api/voice-profiles", tags=["voice-profiles"])

# Rate limiter for voice profiles API
limiter = Limiter(key_func=get_remote_address)


# Models
class VoiceProfileCreate(BaseModel):
    """Voice profile creation request"""
    name: str
    description: Optional[str] = None
    reference_text: Optional[str] = None


class VoiceProfileResponse(BaseModel):
    """Voice profile response model"""
    id: str
    name: str
    description: Optional[str]
    reference_text: Optional[str]
    audio_path: str
    file_size: int
    created_at: datetime
    updated_at: datetime


class VoiceProfileList(BaseModel):
    """Voice profiles list response"""
    profiles: List[VoiceProfileResponse]
    total: int


# Utility Functions
def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file format and size with enhanced security

    Args:
        file: Uploaded audio file

    Raises:
        HTTPException: If file is invalid or potentially malicious
    """
    # Check filename is provided
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    # Sanitize filename - prevent path traversal
    import re
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Path traversal detected."
        )

    # Check file size (max 10MB) - strict enforcement
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail="File size too large. Maximum 10MB allowed."
        )

    # Minimum file size check (audio should be at least 1KB)
    if file.size and file.size < 1024:
        raise HTTPException(
            status_code=400,
            detail="File too small. Minimum 1KB required for valid audio."
        )

    # Check file extension - strict whitelist
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    file_ext = Path(safe_filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )

    # Check MIME type - strict validation
    allowed_mime_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/m4a',
        'audio/flac',
        'audio/ogg', 'audio/vorbis'
        # Removed 'application/octet-stream' for security
    }

    if not file.content_type or file.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or missing MIME type. Expected audio/* types."
        )

    logger.info(
        "voice_profile.file_validated",
        filename=safe_filename,
        size=file.size,
        mime_type=file.content_type
    )


async def save_audio_file(file: UploadFile, profile_id: str) -> Path:
    """
    Save uploaded audio file to voice profiles directory with security checks

    Args:
        file: Uploaded audio file
        profile_id: Unique profile identifier

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file cannot be saved securely
    """
    import re

    # Validate profile_id format (UUID only)
    if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', profile_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid profile ID format"
        )

    # Sanitize filename
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename or "audio.wav")
    file_ext = Path(safe_filename).suffix.lower()

    # Create profile directory with secure path
    profile_dir = config.AUDIO_PROFILES / profile_id

    # Ensure profile_dir is within AUDIO_PROFILES (prevent directory traversal)
    try:
        profile_dir = profile_dir.resolve()
        audio_profiles_real = config.AUDIO_PROFILES.resolve()
        if not str(profile_dir).startswith(str(audio_profiles_real)):
            raise HTTPException(
                status_code=400,
                detail="Invalid profile directory path"
            )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid profile directory"
        )

    profile_dir.mkdir(parents=True, exist_ok=True)

    # Generate secure filename
    audio_path = profile_dir / f"reference{file_ext}"

    # Read and validate file content
    try:
        content = await file.read()

        # Basic audio file header validation
        if file_ext == '.wav' and not content.startswith(b'RIFF'):
            raise HTTPException(
                status_code=400,
                detail="Invalid WAV file format"
            )
        elif file_ext == '.mp3' and not (content.startswith(b'ID3') or content.startswith(b'\xff\xfb')):
            raise HTTPException(
                status_code=400,
                detail="Invalid MP3 file format"
            )

        # Save file with secure permissions
        with open(audio_path, "wb") as buffer:
            buffer.write(content)

        # Set restrictive file permissions (owner read/write only)
        audio_path.chmod(0o600)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("voice_profile.save_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to save audio file"
        )

    logger.info(
        "voice_profile.audio_saved",
        profile_id=profile_id,
        path=str(audio_path),
        size=audio_path.stat().st_size
    )

    return audio_path


async def save_reference_text(profile_dir: Path, reference_text: str):
    """
    Save reference text file for voice profile

    Args:
        profile_dir: Profile directory path
        reference_text: Reference text content
    """
    text_path = profile_dir / "reference.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(reference_text)

    logger.info(
        "voice_profile.text_saved",
        path=str(text_path),
        text_length=len(reference_text)
    )


# API Endpoints
@router.post("", response_model=VoiceProfileResponse)
@limiter.limit("5/minute")  # Limited for file upload
async def create_voice_profile(
    request: Request,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    db = Depends(get_database_service),
    authenticated: bool = Depends(verify_api_token)
):
    """
    Create a new voice profile

    Args:
        name: Profile name
        description: Optional description
        reference_text: Text that matches the audio sample
        audio_file: Audio file for voice cloning
        db: Database service

    Returns:
        Created voice profile
    """
    logger.info("voice_profile.create_start", name=name)

    try:
        # Validate and sanitize inputs
        name = validate_input_string(name, "name", max_length=100)
        if description:
            description = validate_input_string(description, "description", max_length=500)
        if reference_text:
            reference_text = validate_input_string(reference_text, "reference_text", max_length=1000)

        # Validate audio file
        validate_audio_file(audio_file)

        # Generate unique profile ID
        profile_id = str(uuid4())

        # Save audio file
        audio_path = await save_audio_file(audio_file, profile_id)

        # Save reference text file
        profile_dir = config.AUDIO_PROFILES / profile_id
        if reference_text:
            await save_reference_text(profile_dir, reference_text)

        # Create database record
        profile_data = {
            'id': profile_id,
            'name': name,
            'description': description,
            'reference_text': reference_text,
            'audio_path': str(audio_path),
            'file_size': audio_path.stat().st_size,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        await db.create_voice_profile_v2(profile_data)

        logger.info(
            "voice_profile.created",
            profile_id=profile_id,
            name=name,
            file_size=profile_data['file_size']
        )

        return VoiceProfileResponse(**profile_data)

    except HTTPException:
        # Re-raise HTTP exceptions as-is (they're safe)
        raise
    except Exception as e:
        logger.error("voice_profile.create_failed", error=str(e), name=name)

        # Cleanup on failure
        try:
            profile_dir = config.AUDIO_PROFILES / profile_id
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
        except Exception:
            # Silent cleanup failure - don't expose details
            pass

        # Return safe error message
        safe_detail = safe_error_response(str(e), include_detail=False)
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("", response_model=VoiceProfileList)
@limiter.limit("20/minute")
async def list_voice_profiles(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    db = Depends(get_database_service)
):
    """
    List all voice profiles with pagination

    Args:
        limit: Maximum number of profiles to return
        offset: Number of profiles to skip
        db: Database service

    Returns:
        List of voice profiles
    """
    logger.info("voice_profile.list_start", limit=limit, offset=offset)

    try:
        # Get profiles from database
        profiles = await db.get_voice_profiles_v2(limit=limit, offset=offset)
        total = await db.count_voice_profiles_v2()

        # Convert to response models
        profile_responses = [
            VoiceProfileResponse(**profile) for profile in profiles
        ]

        logger.info(
            "voice_profile.list_complete",
            returned=len(profiles),
            total=total
        )

        return VoiceProfileList(profiles=profile_responses, total=total)

    except Exception as e:
        logger.error("voice_profile.list_failed", error=str(e))
        safe_detail = safe_error_response(str(e), include_detail=False)
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("/{profile_id}", response_model=VoiceProfileResponse)
async def get_voice_profile(
    profile_id: str,
    db = Depends(get_database_service)
):
    """
    Get a specific voice profile by ID

    Args:
        profile_id: Voice profile identifier
        db: Database service

    Returns:
        Voice profile details
    """
    logger.info("voice_profile.get_start", profile_id=profile_id)

    try:
        profile = await db.get_voice_profile_v2(profile_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile {profile_id} not found"
            )

        logger.info("voice_profile.get_complete", profile_id=profile_id)
        return VoiceProfileResponse(**profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("voice_profile.get_failed", error=str(e), profile_id=profile_id)
        safe_detail = safe_error_response(str(e), include_detail=False)
        raise HTTPException(status_code=500, detail=safe_detail)


@router.get("/{profile_id}/audio")
async def get_voice_profile_audio(
    profile_id: str,
    db = Depends(get_database_service)
):
    """
    Download voice profile audio file

    Args:
        profile_id: Voice profile identifier
        db: Database service

    Returns:
        Audio file response
    """
    logger.info("voice_profile.download_start", profile_id=profile_id)

    try:
        profile = await db.get_voice_profile_v2(profile_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile {profile_id} not found"
            )

        audio_path = Path(profile['audio_path'])
        if not audio_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Audio file not found"
            )

        logger.info(
            "voice_profile.download_complete",
            profile_id=profile_id,
            file_size=audio_path.stat().st_size
        )

        return FileResponse(
            path=str(audio_path),
            media_type='application/octet-stream',
            filename=f"{profile['name']}_reference{audio_path.suffix}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("voice_profile.download_failed", error=str(e), profile_id=profile_id)
        raise HTTPException(status_code=500, detail="Failed to download audio file")


@router.put("/{profile_id}", response_model=VoiceProfileResponse)
async def update_voice_profile(
    profile_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    db = Depends(get_database_service),
    authenticated: bool = Depends(verify_api_token)
):
    """
    Update an existing voice profile

    Args:
        profile_id: Voice profile identifier
        name: New profile name
        description: New description
        reference_text: New reference text
        audio_file: New audio file (optional)
        db: Database service

    Returns:
        Updated voice profile
    """
    logger.info("voice_profile.update_start", profile_id=profile_id)

    try:
        # Check if profile exists
        existing_profile = await db.get_voice_profile(profile_id)
        if not existing_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile {profile_id} not found"
            )

        update_data = {'updated_at': datetime.utcnow()}

        # Update text fields
        if name is not None:
            update_data['name'] = name
        if description is not None:
            update_data['description'] = description
        if reference_text is not None:
            update_data['reference_text'] = reference_text

        # Update audio file if provided
        if audio_file and audio_file.filename:
            validate_audio_file(audio_file)

            # Remove old audio file
            old_audio_path = Path(existing_profile['audio_path'])
            if old_audio_path.exists():
                old_audio_path.unlink()

            # Save new audio file
            new_audio_path = await save_audio_file(audio_file, profile_id)
            update_data['audio_path'] = str(new_audio_path)
            update_data['file_size'] = new_audio_path.stat().st_size

        # Update database
        await db.update_voice_profile_v2(profile_id, update_data)

        # Get updated profile
        updated_profile = await db.get_voice_profile_v2(profile_id)

        logger.info("voice_profile.updated", profile_id=profile_id)
        return VoiceProfileResponse(**updated_profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("voice_profile.update_failed", error=str(e), profile_id=profile_id)
        safe_detail = safe_error_response(str(e), include_detail=False)
        raise HTTPException(status_code=500, detail=safe_detail)


@router.delete("/{profile_id}")
async def delete_voice_profile(
    profile_id: str,
    db = Depends(get_database_service),
    authenticated: bool = Depends(verify_api_token)
):
    """
    Delete a voice profile and its associated files

    Args:
        profile_id: Voice profile identifier
        db: Database service

    Returns:
        Success message
    """
    logger.info("voice_profile.delete_start", profile_id=profile_id)

    try:
        # Check if profile exists
        profile = await db.get_voice_profile_v2(profile_id)
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile {profile_id} not found"
            )

        # Delete audio files
        profile_dir = config.AUDIO_PROFILES / profile_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

        # Delete from database
        await db.delete_voice_profile_v2(profile_id)

        logger.info("voice_profile.deleted", profile_id=profile_id)
        return {"message": f"Voice profile {profile_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("voice_profile.delete_failed", error=str(e), profile_id=profile_id)
        safe_detail = safe_error_response(str(e), include_detail=False)
        raise HTTPException(status_code=500, detail=safe_detail)


@router.post("/{profile_id}/test")
@limiter.limit("3/minute")  # Limited for synthesis
async def test_voice_profile(
    request: Request,
    profile_id: str,
    text: str = Form(..., max_length=500),
    db = Depends(get_database_service),
    authenticated: bool = Depends(verify_api_token)
):
    """
    Test voice profile by generating a sample audio

    Args:
        profile_id: Voice profile identifier
        text: Text to synthesize
        db: Database service

    Returns:
        Generated audio file
    """
    logger.info("voice_profile.test_start", profile_id=profile_id, text_length=len(text))

    try:
        # Check if profile exists
        profile = await db.get_voice_profile_v2(profile_id)
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile {profile_id} not found"
            )

        # Generate test audio using TTS service
        from avatar.services.tts import get_tts_service

        tts_service = await get_tts_service()

        # Create output path for test audio
        test_output = config.AUDIO_TTS_FAST / f"test_{profile_id}_{uuid4().hex[:8]}.wav"

        # Synthesize using voice profile
        result_path = await tts_service.synthesize_fast(
            text=text,
            voice_profile_name=profile_id,
            output_path=test_output
        )

        logger.info(
            "voice_profile.test_complete",
            profile_id=profile_id,
            output_size=result_path.stat().st_size
        )

        return FileResponse(
            path=str(result_path),
            media_type='audio/wav',
            filename=f"test_{profile['name']}.wav"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("voice_profile.test_failed", error=str(e), profile_id=profile_id)
        raise HTTPException(status_code=500, detail="Failed to generate test audio")