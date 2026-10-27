import base64
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import settings
from backend.main import app
from backend.routers.assistant import get_speech_service
from backend.services.speech import (
    DEFAULT_VOICE,
    MAX_TEXT_LENGTH,
    SpeechConfigurationError,
    SpeechEmptyResponseError,
    SpeechRateLimitError,
    SpeechService,
    SpeechServiceError,
    SpeechTimeoutError,
)

FAKE_AUDIO_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
FAKE_BASE64_AUDIO = base64.b64encode(FAKE_AUDIO_BYTES).decode("utf-8")


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_speech_service():
    mock = MagicMock(spec=SpeechService)
    mock.generate_speech = AsyncMock(return_value=(FAKE_AUDIO_BYTES, "audio/wav"))
    app.dependency_overrides[get_speech_service] = lambda: mock
    return mock


@pytest.fixture
def http_client(mock_speech_service):
    return TestClient(app)


# ── 1. Successful TTS Generation ─────────────────────────────────────────────

def test_speak_success(http_client, mock_speech_service):
    resp = http_client.post(
        "/api/v1/assistant/speak",
        json={"text": "Hello, I am Saarthi. How can I assist you today?"},
    )
    assert resp.status_code == 200
    assert resp.content == FAKE_AUDIO_BYTES
    assert "audio/wav" in resp.headers.get("content-type", "")
    assert resp.headers.get("content-disposition") == "inline"
    mock_speech_service.generate_speech.assert_awaited_once_with(
        text="Hello, I am Saarthi. How can I assist you today?"
    )


# ── 2. Validation: Blank, Empty, and Missing Text ────────────────────────────

def test_speak_blank_text_returns_422(http_client):
    resp = http_client.post("/api/v1/assistant/speak", json={"text": "   "})
    assert resp.status_code == 422
    assert "text cannot be empty" in resp.json()["detail"]


def test_speak_empty_text_returns_422(http_client):
    resp = http_client.post("/api/v1/assistant/speak", json={"text": ""})
    assert resp.status_code == 422


def test_speak_missing_text_returns_422(http_client):
    resp = http_client.post("/api/v1/assistant/speak", json={})
    assert resp.status_code == 422


# ── 3. Validation: Excessively Long Text ──────────────────────────────────────

def test_speak_excessively_long_text_returns_422(http_client):
    long_text = "A" * (MAX_TEXT_LENGTH + 1)
    resp = http_client.post("/api/v1/assistant/speak", json={"text": long_text})
    assert resp.status_code == 422
    assert "exceeds maximum allowed length" in resp.json()["detail"]


# ── 4. Error Handling: Missing Credentials (503) ─────────────────────────────

def test_speak_missing_credentials_returns_503(http_client, mock_speech_service):
    mock_speech_service.generate_speech.side_effect = SpeechConfigurationError("API key missing")
    resp = http_client.post("/api/v1/assistant/speak", json={"text": "Hello"})
    assert resp.status_code == 503
    assert "AI credentials are not configured" in resp.json()["detail"]


# ── 5. Error Handling: Rate Limit Exceeded (429) ─────────────────────────────

def test_speak_rate_limit_returns_429(http_client, mock_speech_service):
    mock_speech_service.generate_speech.side_effect = SpeechRateLimitError("Quota exceeded")
    resp = http_client.post("/api/v1/assistant/speak", json={"text": "Hello"})
    assert resp.status_code == 429
    assert "rate limit was exceeded" in resp.json()["detail"]


# ── 6. Error Handling: Gateway Timeout (504) ─────────────────────────────────

def test_speak_timeout_returns_504(http_client, mock_speech_service):
    mock_speech_service.generate_speech.side_effect = SpeechTimeoutError("Request timed out")
    resp = http_client.post("/api/v1/assistant/speak", json={"text": "Hello"})
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]


# ── 7. Error Handling: Empty Response (502) ──────────────────────────────────

def test_speak_empty_response_returns_502(http_client, mock_speech_service):
    mock_speech_service.generate_speech.side_effect = SpeechEmptyResponseError("Empty audio")
    resp = http_client.post("/api/v1/assistant/speak", json={"text": "Hello"})
    assert resp.status_code == 502
    assert "empty response" in resp.json()["detail"]


# ── 8. Error Handling: Provider Failure (502) ────────────────────────────────

def test_speak_provider_error_returns_502(http_client, mock_speech_service):
    mock_speech_service.generate_speech.side_effect = SpeechServiceError("Interactions API failure")
    resp = http_client.post("/api/v1/assistant/speak", json={"text": "Hello"})
    assert resp.status_code == 502
    assert "could not process" in resp.json()["detail"]


# ── 9. SpeechService Unit Tests ──────────────────────────────────────────────

def test_speech_service_defaults():
    service = SpeechService()
    assert service.model == settings.gemini_tts_model
    assert service.voice == DEFAULT_VOICE


@pytest.mark.anyio
async def test_speech_service_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
    service = SpeechService()
    with pytest.raises(SpeechConfigurationError):
        await service.generate_speech("Test text")


@pytest.mark.anyio
async def test_speech_service_successful_generation():
    mock_interaction = MagicMock()
    mock_output_audio = MagicMock()
    mock_output_audio.data = FAKE_BASE64_AUDIO
    mock_output_audio.mime_type = "audio/wav"
    mock_interaction.output_audio = mock_output_audio

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(return_value=mock_interaction)

    service = SpeechService(client=mock_client)
    audio_bytes, mime_type = await service.generate_speech("Hello test", voice="Kore")

    assert audio_bytes == FAKE_AUDIO_BYTES
    assert mime_type == "audio/wav"

    mock_client.aio.interactions.create.assert_awaited_once()
    call_kwargs = mock_client.aio.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == settings.gemini_tts_model
    assert call_kwargs["response_format"] == {"type": "audio"}
    assert call_kwargs["generation_config"]["speech_config"] == [{"voice": "Kore"}]


@pytest.mark.anyio
async def test_speech_service_fallback_steps_extraction():
    mock_interaction = MagicMock()
    mock_interaction.output_audio = None

    mock_step = MagicMock()
    mock_output_audio = MagicMock()
    mock_output_audio.data = FAKE_BASE64_AUDIO
    mock_output_audio.mime_type = "audio/mp3"
    mock_step.output_audio = mock_output_audio
    mock_interaction.steps = [mock_step]

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(return_value=mock_interaction)

    service = SpeechService(client=mock_client)
    audio_bytes, mime_type = await service.generate_speech("Step audio test")

    assert audio_bytes == FAKE_AUDIO_BYTES
    assert mime_type == "audio/mp3"
