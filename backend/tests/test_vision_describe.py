import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app
from backend.routers.vision import get_vision_client
from backend.services.vision_model import (
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)

# Minimal valid file headers (magic bytes only)
JPEG_HEADER = b"\xff\xd8\xff" + b"\x00" * 20
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
GARBAGE_BYTES = b"\x00\x01\x02\x03" * 10


def make_mock_client(description: str = "A chair is beside a table.") -> MagicMock:
    mock = MagicMock()
    mock.generate_vision = AsyncMock(return_value=description)
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


# ── 1. Successful image upload ────────────────────────────────────────────────

def test_describe_jpeg_success(http_client):
    mock = make_mock_client("A desk with a laptop on it.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"description": "A desk with a laptop on it."}


def test_describe_png_success(http_client):
    mock = make_mock_client("A kitchen with a table.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.png", PNG_HEADER, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "A kitchen with a table."


def test_describe_webp_success(http_client):
    mock = make_mock_client("An outdoor scene with trees.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.webp", WEBP_HEADER, "image/webp")},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "An outdoor scene with trees."


# ── 2. Optional user question forwarded to AI client ─────────────────────────

def test_user_question_forwarded(http_client):
    mock = make_mock_client("The doorway is one metre ahead.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
        data={"question": "Is there a doorway?"},
    )
    assert resp.status_code == 200
    mock.generate_vision.assert_awaited_once()
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["prompt"] == "Is there a doorway?"


# ── 3. Default instruction used when no question supplied ────────────────────

def test_default_prompt_used_when_no_question(http_client):
    from backend.routers.vision import DEFAULT_QUESTION

    mock = make_mock_client("A room with furniture.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 200
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["prompt"] == DEFAULT_QUESTION


def test_blank_question_falls_back_to_default(http_client):
    from backend.routers.vision import DEFAULT_QUESTION

    mock = make_mock_client("A room with furniture.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
        data={"question": "   "},
    )
    assert resp.status_code == 200
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["prompt"] == DEFAULT_QUESTION


# ── 4. Missing / empty image ──────────────────────────────────────────────────

def test_missing_image_field_returns_422(http_client):
    resp = http_client.post("/api/v1/vision/describe", data={"question": "hello"})
    assert resp.status_code == 422


def test_empty_image_bytes_returns_422(http_client):
    mock = make_mock_client()
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422
    mock.generate_vision.assert_not_awaited()


# ── 5. Unsupported media type ─────────────────────────────────────────────────

def test_unsupported_mime_type_returns_415(http_client):
    mock = make_mock_client()
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("doc.pdf", b"%PDF-1.4" + b"\x00" * 20, "application/pdf")},
    )
    assert resp.status_code == 415
    mock.generate_vision.assert_not_awaited()


def test_mismatched_file_signature_returns_415(http_client):
    """JPEG MIME declared but content is random bytes — signature check should reject it."""
    mock = make_mock_client()
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("sneaky.jpg", GARBAGE_BYTES, "image/jpeg")},
    )
    assert resp.status_code == 415
    mock.generate_vision.assert_not_awaited()


# ── 6. Upload-size limit ──────────────────────────────────────────────────────

def test_oversized_image_returns_413(http_client):
    from backend.routers.vision import MAX_IMAGE_SIZE

    mock = make_mock_client()
    override_client(mock)

    oversized = JPEG_HEADER + b"\x00" * (MAX_IMAGE_SIZE + 1)
    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("big.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413
    mock.generate_vision.assert_not_awaited()


# ── 7. Missing AI credentials ─────────────────────────────────────────────────

def test_missing_credentials_returns_503(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelConfigurationError("No API key")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "credentials" in resp.json()["detail"].lower()


# ── 8. Model provider errors and empty responses ──────────────────────────────

def test_model_error_returns_502(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelError("Provider error")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 502


def test_empty_model_response_returns_502(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelEmptyResponseError("Empty response")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("photo.jpg", JPEG_HEADER, "image/jpeg")},
    )
    assert resp.status_code == 502


# ── 9. Regression: existing tests still accessible ───────────────────────────
# (Covered by running the full suite; this file does not import test_health or
# test_config — pytest discovers them separately.)
