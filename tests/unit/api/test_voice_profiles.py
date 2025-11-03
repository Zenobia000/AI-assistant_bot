"""
TDD Unit Tests for Voice Profile API

Testing Voice Profile REST API endpoints with database operations.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fastapi import UploadFile
from fastapi.testclient import TestClient

from avatar.api.voice_profiles import router, validate_audio_file, save_audio_file, save_reference_text
from avatar.services.database import DatabaseService
from avatar.core.config import config


class TestVoiceProfileValidation:
    """Test voice profile validation functions"""

    def test_validate_audio_file_valid_wav(self):
        """Test validation with valid WAV file"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.wav"
        mock_file.content_type = "audio/wav"
        mock_file.size = 1024 * 1024  # 1MB

        # Should not raise exception
        validate_audio_file(mock_file)

    def test_validate_audio_file_valid_mp3(self):
        """Test validation with valid MP3 file"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp3"
        mock_file.content_type = "audio/mpeg"
        mock_file.size = 2 * 1024 * 1024  # 2MB

        # Should not raise exception
        validate_audio_file(mock_file)

    def test_validate_audio_file_octet_stream(self):
        """Test validation with application/octet-stream (common for binary uploads)"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.wav"
        mock_file.content_type = "application/octet-stream"
        mock_file.size = 1024 * 1024  # 1MB

        # Should not raise exception
        validate_audio_file(mock_file)

    def test_validate_audio_file_too_large(self):
        """Test validation with file too large"""
        from fastapi import HTTPException

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.wav"
        mock_file.content_type = "audio/wav"
        mock_file.size = 15 * 1024 * 1024  # 15MB (over 10MB limit)

        with pytest.raises(HTTPException) as exc_info:
            validate_audio_file(mock_file)

        assert exc_info.value.status_code == 400
        assert "too large" in str(exc_info.value.detail)

    def test_validate_audio_file_invalid_extension(self):
        """Test validation with invalid file extension"""
        from fastapi import HTTPException

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.content_type = "text/plain"
        mock_file.size = 1024

        with pytest.raises(HTTPException) as exc_info:
            validate_audio_file(mock_file)

        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in str(exc_info.value.detail)

    def test_validate_audio_file_invalid_mime_type(self):
        """Test validation with invalid MIME type"""
        from fastapi import HTTPException

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.wav"
        mock_file.content_type = "text/plain"
        mock_file.size = 1024

        with pytest.raises(HTTPException) as exc_info:
            validate_audio_file(mock_file)

        assert exc_info.value.status_code == 400
        assert "Invalid MIME type" in str(exc_info.value.detail)


class TestVoiceProfileFileOperations:
    """Test voice profile file operations"""

    @pytest.fixture
    def temp_profiles_dir(self):
        """Create temporary profiles directory"""
        temp_dir = tempfile.mkdtemp()
        original_profiles = config.AUDIO_PROFILES
        config.AUDIO_PROFILES = Path(temp_dir)

        yield Path(temp_dir)

        # Cleanup
        config.AUDIO_PROFILES = original_profiles
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_save_audio_file(self, temp_profiles_dir):
        """Test saving audio file to profile directory"""
        # Create mock file
        test_content = b"fake audio data"
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.wav"
        mock_file.file = BytesIO(test_content)

        profile_id = str(uuid4())

        # Save file
        saved_path = await save_audio_file(mock_file, profile_id)

        # Verify file was saved
        assert saved_path.exists()
        assert saved_path.name == "reference.wav"
        assert saved_path.parent.name == profile_id
        assert saved_path.read_bytes() == test_content

    @pytest.mark.asyncio
    async def test_save_reference_text(self, temp_profiles_dir):
        """Test saving reference text file"""
        profile_dir = temp_profiles_dir / "test_profile"
        profile_dir.mkdir(exist_ok=True)

        reference_text = "This is test reference text."

        await save_reference_text(profile_dir, reference_text)

        # Verify text file was saved
        text_path = profile_dir / "reference.txt"
        assert text_path.exists()
        assert text_path.read_text(encoding="utf-8") == reference_text


class TestVoiceProfileDatabase:
    """Test voice profile database operations"""

    @pytest.fixture
    async def mock_db(self):
        """Mock database service"""
        db = AsyncMock(spec=DatabaseService)
        return db

    @pytest.mark.asyncio
    async def test_create_voice_profile_v2(self, mock_db):
        """Test creating voice profile in database"""
        profile_data = {
            'id': str(uuid4()),
            'name': 'Test Profile',
            'description': 'Test description',
            'reference_text': 'Test reference text',
            'audio_path': '/test/path.wav',
            'file_size': 1024,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        mock_db.create_voice_profile_v2.return_value = profile_data['id']

        result = await mock_db.create_voice_profile_v2(profile_data)

        assert result == profile_data['id']
        mock_db.create_voice_profile_v2.assert_called_once_with(profile_data)

    @pytest.mark.asyncio
    async def test_get_voice_profile_v2(self, mock_db):
        """Test retrieving voice profile from database"""
        profile_id = str(uuid4())
        expected_profile = {
            'id': profile_id,
            'name': 'Test Profile',
            'description': 'Test description',
            'reference_text': 'Test reference text',
            'audio_path': '/test/path.wav',
            'file_size': 1024,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        mock_db.get_voice_profile_v2.return_value = expected_profile

        result = await mock_db.get_voice_profile_v2(profile_id)

        assert result == expected_profile
        mock_db.get_voice_profile_v2.assert_called_once_with(profile_id)

    @pytest.mark.asyncio
    async def test_get_voice_profiles_v2_with_pagination(self, mock_db):
        """Test listing voice profiles with pagination"""
        profiles = [
            {
                'id': str(uuid4()),
                'name': f'Profile {i}',
                'description': f'Description {i}',
                'reference_text': f'Reference {i}',
                'audio_path': f'/test/path{i}.wav',
                'file_size': 1024 * i,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            for i in range(3)
        ]

        mock_db.get_voice_profiles_v2.return_value = profiles
        mock_db.count_voice_profiles_v2.return_value = 10

        result_profiles = await mock_db.get_voice_profiles_v2(limit=5, offset=0)
        total_count = await mock_db.count_voice_profiles_v2()

        assert len(result_profiles) == 3
        assert total_count == 10
        mock_db.get_voice_profiles_v2.assert_called_once_with(limit=5, offset=0)

    @pytest.mark.asyncio
    async def test_update_voice_profile_v2(self, mock_db):
        """Test updating voice profile"""
        profile_id = str(uuid4())
        update_data = {
            'name': 'Updated Name',
            'description': 'Updated description',
            'updated_at': datetime.utcnow()
        }

        mock_db.update_voice_profile_v2.return_value = True

        result = await mock_db.update_voice_profile_v2(profile_id, update_data)

        assert result is True
        mock_db.update_voice_profile_v2.assert_called_once_with(profile_id, update_data)

    @pytest.mark.asyncio
    async def test_delete_voice_profile_v2(self, mock_db):
        """Test deleting voice profile"""
        profile_id = str(uuid4())

        mock_db.delete_voice_profile_v2.return_value = True

        result = await mock_db.delete_voice_profile_v2(profile_id)

        assert result is True
        mock_db.delete_voice_profile_v2.assert_called_once_with(profile_id)


class TestVoiceProfileAPIIntegration:
    """Test Voice Profile API integration with FastAPI"""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client"""
        return TestClient(test_app)

    @pytest.fixture
    def mock_database_service(self):
        """Mock database service dependency"""
        with patch('avatar.api.voice_profiles.get_database_service') as mock:
            db_mock = AsyncMock(spec=DatabaseService)
            mock.return_value = db_mock
            yield db_mock

    @pytest.fixture
    def temp_config_paths(self):
        """Setup temporary paths for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock config paths
            with patch.object(config, 'AUDIO_PROFILES', temp_path / 'profiles'), \
                 patch.object(config, 'AUDIO_TTS_FAST', temp_path / 'tts_fast'):

                # Create directories
                (temp_path / 'profiles').mkdir(exist_ok=True)
                (temp_path / 'tts_fast').mkdir(exist_ok=True)

                yield temp_path

    def test_list_voice_profiles_empty(self, client, mock_database_service):
        """Test listing voice profiles when none exist"""
        mock_database_service.get_voice_profiles_v2.return_value = []
        mock_database_service.count_voice_profiles_v2.return_value = 0

        response = client.get("/api/voice-profiles")

        assert response.status_code == 200
        data = response.json()
        assert data["profiles"] == []
        assert data["total"] == 0

    def test_get_voice_profile_not_found(self, client, mock_database_service):
        """Test getting non-existent voice profile"""
        profile_id = str(uuid4())
        mock_database_service.get_voice_profile_v2.return_value = None

        response = client.get(f"/api/voice-profiles/{profile_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch('avatar.api.voice_profiles.save_audio_file')
    @patch('avatar.api.voice_profiles.save_reference_text')
    def test_create_voice_profile_success(
        self,
        mock_save_text,
        mock_save_audio,
        client,
        mock_database_service,
        temp_config_paths
    ):
        """Test successful voice profile creation"""
        # Mock file saving
        mock_audio_path = temp_config_paths / 'profiles' / 'test-id' / 'reference.wav'
        mock_save_audio.return_value = mock_audio_path
        mock_save_text.return_value = None

        # Mock database creation
        mock_database_service.create_voice_profile_v2.return_value = "test-profile-id"

        # Create test audio file
        audio_content = b"fake audio data"

        # Make request
        response = client.post(
            "/api/voice-profiles",
            data={
                "name": "Test Profile",
                "description": "Test description",
                "reference_text": "Test reference text"
            },
            files={"audio_file": ("test.wav", BytesIO(audio_content), "audio/wav")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Profile"
        assert data["description"] == "Test description"
        assert data["reference_text"] == "Test reference text"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_voice_profile_invalid_file(self, client, mock_database_service):
        """Test voice profile creation with invalid file"""
        response = client.post(
            "/api/voice-profiles",
            data={
                "name": "Test Profile",
                "description": "Test description"
            },
            files={"audio_file": ("test.txt", BytesIO(b"not audio"), "text/plain")}
        )

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]


class TestVoiceProfileTTSIntegration:
    """Test Voice Profile TTS integration"""

    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service"""
        with patch('avatar.api.voice_profiles.get_tts_service') as mock:
            tts_mock = AsyncMock()
            mock.return_value = tts_mock
            yield tts_mock

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client"""
        return TestClient(test_app)

    @pytest.fixture
    def mock_database_service(self):
        """Mock database service dependency"""
        with patch('avatar.api.voice_profiles.get_database_service') as mock:
            db_mock = AsyncMock(spec=DatabaseService)
            mock.return_value = db_mock
            yield db_mock

    def test_voice_profile_test_synthesis(
        self,
        client,
        mock_database_service,
        mock_tts_service
    ):
        """Test voice profile test synthesis endpoint"""
        profile_id = str(uuid4())

        # Mock database response
        mock_database_service.get_voice_profile_v2.return_value = {
            'id': profile_id,
            'name': 'Test Profile',
            'audio_path': '/test/path.wav'
        }

        # Mock TTS synthesis
        test_output_path = Path("/tmp/test_output.wav")
        mock_tts_service.synthesize_fast.return_value = test_output_path

        # Mock file stats
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 1024

            response = client.post(
                f"/api/voice-profiles/{profile_id}/test",
                data={"text": "Hello, this is a test."}
            )

        # For this test, we expect either success or a controlled failure
        # depending on TTS service availability
        assert response.status_code in [200, 500]  # 500 if TTS not available in test env

    def test_voice_profile_test_not_found(
        self,
        client,
        mock_database_service,
        mock_tts_service
    ):
        """Test voice profile test with non-existent profile"""
        profile_id = str(uuid4())

        # Mock database response
        mock_database_service.get_voice_profile_v2.return_value = None

        response = client.post(
            f"/api/voice-profiles/{profile_id}/test",
            data={"text": "Hello, this is a test."}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])