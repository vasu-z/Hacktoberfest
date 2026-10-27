import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app
from backend.routers.vision import READ_TEXT_PROMPT, get_vision_client
from backend.services.vision_model import (
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)

JPEG_HEADER = b"\xff\xd8\xff" + b"\x00" * 20
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
GARBAGE_BYTES = b"\x00\x01\x02\x03" * 10


def make_mock_client(text: str = "EMERGENCY EXIT") -> MagicMock:
    mock = MagicMock()
    mock.generate_vision = AsyncMock(return_value=text)
    return mock


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def override_client(mock):
    app.dependency_overrides[get_vision_client] = lambda: mock


@pytest.fixture
def http_client():
    return TestClient(app)


# ── 1. Successful transcription response ─────────────────────────────────────

def test_read_text_jpeg_success(http_client):
    mock = make_mock_client("EXIT")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "EXIT"}


def test_read_text_png_success(http_client):
    mock = make_mock_client("ROOM 402")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.png", PNG_HEADER, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "ROOM 402"}


def test_read_text_webp_success(http_client):
    mock = make_mock_client("CAUTION: WET FLOOR")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.webp", WEBP_HEADER, "image/webp")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "CAUTION: WET FLOOR"}


# ── 2. Model receives appropriate transcription-focused prompt ───────────────

def test_read_text_receives_transcription_prompt(http_client):
    mock = make_mock_client("PUSH TO OPEN")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("door.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 200
    mock.generate_vision.assert_awaited_once()
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["prompt"] == READ_TEXT_PROMPT
    assert "transcribe" in READ_TEXT_PROMPT.lower()
    assert "reading order" in READ_TEXT_PROMPT.lower()
    assert "unreadable" in READ_TEXT_PROMPT.lower()


# ── 3. Correct image bytes and MIME type reach model client ──────────────────

def test_read_text_forwards_correct_bytes_and_mime(http_client):
    mock = make_mock_client("SAMPLE TEXT")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sample.png", PNG_HEADER, "image/png")},
    )
    assert resp.status_code == 200
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["image_bytes"] == PNG_HEADER
    assert call_kwargs["mime_type"] == "image/png"


# ── 4. Empty and unsupported uploads are rejected ────────────────────────────

def test_read_text_missing_image_field_returns_422(http_client):
    resp = http_client.post("/api/v1/vision/read-text")
    assert resp.status_code == 422


def test_read_text_empty_image_bytes_returns_422(http_client):
    mock = make_mock_client()
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()
    mock.generate_vision.assert_not_awaited()


def test_read_text_unsupported_mime_type_returns_415(http_client):
    mock = make_mock_client()
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("doc.pdf", b"%PDF-1.4" + b"\x00" * 20, "application/pdf")},
    )
    assert resp.status_code == 415
    mock.generate_vision.assert_not_awaited()


# ── 5. Invalid image signatures are rejected ──────────────────────────────────

def test_read_text_mismatched_file_signature_returns_415(http_client):
    mock = make_mock_client()
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("fake.jpg", GARBAGE_BYTES, "image/jpeg")},
    )
    assert resp.status_code == 415
    mock.generate_vision.assert_not_awaited()


# ── 6. Oversized images are rejected ──────────────────────────────────────────

def test_read_text_oversized_image_returns_413(http_client):
    from backend.routers.vision import MAX_IMAGE_SIZE

    mock = make_mock_client()
    override_client(mock)

    oversized = JPEG_HEADER + b"\x00" * (MAX_IMAGE_SIZE + 1)
    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413
    mock.generate_vision.assert_not_awaited()


# ── 7. Missing credentials return a controlled error ──────────────────────────

def test_read_text_missing_credentials_returns_503(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelConfigurationError("GEMINI_API_KEY is not configured.")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "credentials" in resp.json()["detail"].lower()


# ── 8. Provider failures and empty responses are handled ─────────────────────

def test_read_text_provider_failure_returns_502(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelError("Upstream connection failure.")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 502


def test_read_text_empty_model_response_returns_502(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelEmptyResponseError("Model returned empty text.")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 502


# ── 9. Scene-understanding endpoint still works ───────────────────────────────

def test_scene_understanding_endpoint_still_works(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(return_value="A well-lit hallway.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"description": "A well-lit hallway."}
