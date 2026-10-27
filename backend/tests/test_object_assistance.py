import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image
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


def _make_test_image(fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (20, 20), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


JPEG_IMAGE = _make_test_image("JPEG")
PNG_IMAGE = _make_test_image("PNG")
WEBP_IMAGE = _make_test_image("WEBP")
GARBAGE_BYTES = b"\x00\x01\x02\x03" * 10


def make_mock_client(model_output: dict | str) -> MagicMock:
    raw_text = json.dumps(model_output) if isinstance(model_output, dict) else model_output
    mock = MagicMock()
    mock.generate_vision = AsyncMock(return_value=raw_text)
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


# ── 1. Clearly visible object ────────────────────────────────────────────────

def test_assist_clearly_visible_object(http_client):
    mock_payload = {
        "found": True,
        "description": "A blue water bottle is standing in the center of the desk next to the laptop.",
        "certainty": "clear",
    }
    mock = make_mock_client(mock_payload)
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("desk.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"target_object": "water bottle"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_object"] == "water bottle"
    assert data["found"] is True
    assert data["certainty"] == "clear"
    assert "water bottle" in data["description"]
    assert "desk" in data["description"]


# ── 2. Object not visible ────────────────────────────────────────────────────

def test_assist_object_not_visible(http_client):
    mock_payload = {
        "found": False,
        "description": "No keys could be identified on the desk or surrounding visible area.",
        "certainty": "not_visible",
    }
    mock = make_mock_client(mock_payload)
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("desk.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"target_object": "keys"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_object"] == "keys"
    assert data["found"] is False
    assert data["certainty"] == "not_visible"
    assert "keys" in data["description"].lower()


# ── 3. Uncertain or partially obscured object ────────────────────────────────

def test_assist_uncertain_or_obscured_object(http_client):
    mock_payload = {
        "found": True,
        "description": "There is a partially obscured black handle beneath the papers that might be scissors.",
        "certainty": "uncertain",
    }
    mock = make_mock_client(mock_payload)
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("clutter.png", PNG_IMAGE, "image/png")},
        data={"target_object": "scissors"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_object"] == "scissors"
    assert data["found"] is True
    assert data["certainty"] == "uncertain"
    assert "obscured" in data["description"]


# ── 4. Target object is passed correctly to model prompt ─────────────────────

def test_assist_passes_target_object_to_prompt(http_client):
    mock = make_mock_client({
        "found": True,
        "description": "A pair of reading glasses is on the nightstand.",
        "certainty": "clear",
    })
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("table.webp", WEBP_IMAGE, "image/webp")},
        data={"target_object": "reading glasses"},
    )
    assert resp.status_code == 200
    mock.generate_vision.assert_awaited_once()
    prompt = mock.generate_vision.call_args.kwargs["prompt"]
    assert "reading glasses" in prompt


# ── 5. Image bytes and MIME type are forwarded correctly ─────────────────────

def test_assist_forwards_image_bytes_and_mime(http_client):
    mock = make_mock_client({
        "found": True,
        "description": "A white mug is on the table.",
        "certainty": "clear",
    })
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("mug.png", PNG_IMAGE, "image/png")},
        data={"target_object": "mug"},
    )
    assert resp.status_code == 200
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["image_bytes"] == PNG_IMAGE
    assert call_kwargs["mime_type"] == "image/png"


# ── 6. Missing or invalid input ──────────────────────────────────────────────

def test_assist_missing_image_returns_422(http_client):
    resp = http_client.post(
        "/api/v1/objects/assist",
        data={"target_object": "chair"},
    )
    assert resp.status_code == 422


def test_assist_missing_target_object_returns_422(http_client):
    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("photo.jpg", JPEG_IMAGE, "image/jpeg")},
    )
    assert resp.status_code == 422


def test_assist_blank_target_object_returns_422(http_client):
    mock = make_mock_client({})
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("photo.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"target_object": "    "},
    )
    assert resp.status_code == 422
    assert "target_object" in resp.json()["detail"].lower()
    mock.generate_vision.assert_not_awaited()


def test_assist_empty_image_bytes_returns_422(http_client):
    mock = make_mock_client({})
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
        data={"target_object": "cup"},
    )
    assert resp.status_code == 422
    mock.generate_vision.assert_not_awaited()


def test_assist_unsupported_mime_type_returns_415(http_client):
    mock = make_mock_client({})
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("doc.pdf", b"%PDF-1.4" + b"\x00" * 20, "application/pdf")},
        data={"target_object": "book"},
    )
    assert resp.status_code == 415
    mock.generate_vision.assert_not_awaited()


def test_assist_mismatched_signature_returns_415(http_client):
    mock = make_mock_client({})
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("fake.jpg", GARBAGE_BYTES, "image/jpeg")},
        data={"target_object": "phone"},
    )
    assert resp.status_code == 415
    mock.generate_vision.assert_not_awaited()


def test_assist_oversized_image_returns_413(http_client):
    from backend.routers.vision import MAX_IMAGE_SIZE

    mock = make_mock_client({})
    override_client(mock)

    oversized = (b"\xff\xd8\xff" + b"\x00" * 20) + b"\x00" * (MAX_IMAGE_SIZE + 1)
    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("huge.jpg", oversized, "image/jpeg")},
        data={"target_object": "laptop"},
    )
    assert resp.status_code == 413
    mock.generate_vision.assert_not_awaited()


# ── 7. Missing credentials and provider errors ───────────────────────────────

def test_assist_missing_credentials_returns_503(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelConfigurationError("GEMINI_API_KEY is not configured.")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("photo.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"target_object": "wallet"},
    )
    assert resp.status_code == 503
    assert "credentials" in resp.json()["detail"].lower()


def test_assist_provider_error_returns_502(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelError("Upstream model timeout.")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("photo.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"target_object": "wallet"},
    )
    assert resp.status_code == 502


def test_assist_empty_model_response_returns_502(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(
        side_effect=VisionModelEmptyResponseError("Empty response.")
    )
    override_client(mock)

    resp = http_client.post(
        "/api/v1/objects/assist",
        files={"image": ("photo.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"target_object": "wallet"},
    )
    assert resp.status_code == 502


# ── 8. Existing scene-understanding and text-reading endpoints still work ────

def test_existing_scene_description_endpoint_works(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(return_value="A living room with a sofa.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("room.jpg", JPEG_IMAGE, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"description": "A living room with a sofa."}


def test_existing_text_reading_endpoint_works(http_client):
    mock = MagicMock()
    mock.generate_vision = AsyncMock(return_value="CAUTION")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.jpg", JPEG_IMAGE, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "CAUTION"}
