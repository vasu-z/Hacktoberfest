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
    img = Image.new("RGB", (30, 30), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


JPEG_IMAGE = _make_test_image("JPEG")
PNG_IMAGE = _make_test_image("PNG")
GARBAGE_BYTES = b"\x00\x01\x02\x03" * 10


def make_mock_client(model_output: dict | str) -> MagicMock:
    raw_text = json.dumps(model_output) if isinstance(model_output, dict) else model_output
    mock = MagicMock()
    mock.generate_text = AsyncMock(return_value=raw_text)
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


# ── 1. Successful structured guidance response (text-only) ───────────────────

def test_guide_task_text_success(http_client):
    mock_payload = {
        "task": "Brew a cup of tea",
        "steps": [
            "Fill kettle with fresh cold water.",
            "Turn on the kettle and wait for it to boil.",
            "Place tea bag in your mug.",
            "Pour hot water carefully into the mug.",
            "Steep for 3 minutes, then remove tea bag.",
        ],
        "current_step": "Fill kettle with fresh cold water.",
        "needs_visual_confirmation": False,
        "cautions": ["Be careful when pouring boiling water to prevent burns."],
    }
    mock = make_mock_client(mock_payload)
    override_client(mock)

    resp = http_client.post(
        "/api/v1/tasks/guide",
        data={"task": "Brew a cup of tea"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "Brew a cup of tea"
    assert len(data["steps"]) == 5
    assert data["current_step"] == "Fill kettle with fresh cold water."
    assert data["needs_visual_confirmation"] is False
    assert len(data["cautions"]) == 1
    mock.generate_text.assert_awaited_once()
    mock.generate_vision.assert_not_awaited()


# ── 2. Missing or blank task input ───────────────────────────────────────────

def test_guide_task_missing_task_returns_422(http_client):
    resp = http_client.post("/api/v1/tasks/guide")
    assert resp.status_code == 422


def test_guide_task_blank_task_returns_422(http_client):
    mock = make_mock_client({})
    override_client(mock)

    resp = http_client.post(
        "/api/v1/tasks/guide",
        data={"task": "     "},
    )
    assert resp.status_code == 422
    assert "task cannot be empty" in resp.json()["detail"].lower()
    mock.generate_text.assert_not_awaited()


# ── 3. Optional context and previous progress included in prompt ─────────────

def test_guide_task_context_and_previous_steps_forwarded(http_client):
    mock_payload = {
        "task": "Bake cookies",
        "steps": ["Shape dough into balls", "Place on baking sheet", "Bake for 10 minutes"],
        "current_step": "Shape dough into balls",
        "needs_visual_confirmation": True,
        "cautions": ["Oven is preheated and hot."],
    }
    mock = make_mock_client(mock_payload)
    override_client(mock)

    resp = http_client.post(
        "/api/v1/tasks/guide",
        data={
            "task": "Bake cookies",
            "context": "Using pre-made chocolate chip dough in a toaster oven.",
            "previous_steps": "Oven preheated to 350F; dough is unrolled.",
        },
    )
    assert resp.status_code == 200
    mock.generate_text.assert_awaited_once()
    prompt = mock.generate_text.call_args.kwargs["prompt"]
    assert "Bake cookies" in prompt
    assert "toaster oven" in prompt
    assert "Oven preheated" in prompt


# ── 4. Image-assisted guidance ───────────────────────────────────────────────

def test_guide_task_with_image_success(http_client):
    mock_payload = {
        "task": "Plug in charging cable",
        "steps": [
            "Locate the charging port on the right edge of the laptop.",
            "Insert the USB-C connector straight into the port.",
        ],
        "current_step": "Locate the charging port on the right edge of the laptop.",
        "needs_visual_confirmation": True,
        "cautions": ["Do not force the connector if resistance is felt."],
    }
    mock = make_mock_client(mock_payload)
    override_client(mock)

    resp = http_client.post(
        "/api/v1/tasks/guide",
        files={"image": ("desk.jpg", JPEG_IMAGE, "image/jpeg")},
        data={"task": "Plug in charging cable"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "Plug in charging cable"
    assert data["needs_visual_confirmation"] is True
    mock.generate_vision.assert_awaited_once()
    mock.generate_text.assert_not_awaited()
    call_kwargs = mock.generate_vision.call_args.kwargs
    assert call_kwargs["mime_type"] == "image/jpeg"
    assert len(call_kwargs["image_bytes"]) > 0


# ── 5. Correct selection of text vs image generation methods ─────────────────

def test_method_selection_without_image_calls_text_generation(http_client):
    mock = make_mock_client({
        "task": "Tie shoes",
        "steps": ["Cross laces", "Pull tight"],
        "current_step": "Cross laces",
        "needs_visual_confirmation": False,
        "cautions": [],
    })
    override_client(mock)

    http_client.post("/api/v1/tasks/guide", data={"task": "Tie shoes"})
    mock.generate_text.assert_awaited_once()
    mock.generate_vision.assert_not_awaited()


def test_method_selection_with_image_calls_vision_generation(http_client):
    mock = make_mock_client({
        "task": "Tie shoes",
        "steps": ["Cross laces", "Pull tight"],
        "current_step": "Cross laces",
        "needs_visual_confirmation": False,
        "cautions": [],
    })
    override_client(mock)

    http_client.post(
        "/api/v1/tasks/guide",
        files={"image": ("shoes.png", PNG_IMAGE, "image/png")},
        data={"task": "Tie shoes"},
    )
    mock.generate_vision.assert_awaited_once()
    mock.generate_text.assert_not_awaited()


# ── 6. Missing credentials and provider failures ─────────────────────────────

def test_guide_task_missing_credentials_returns_503(http_client):
    mock = MagicMock()
    mock.generate_text = AsyncMock(
        side_effect=VisionModelConfigurationError("API key not configured")
    )
    override_client(mock)

    resp = http_client.post("/api/v1/tasks/guide", data={"task": "Fold clothes"})
    assert resp.status_code == 503
    assert "credentials" in resp.json()["detail"].lower()


def test_guide_task_provider_error_returns_502(http_client):
    mock = MagicMock()
    mock.generate_text = AsyncMock(
        side_effect=VisionModelError("Upstream timeout")
    )
    override_client(mock)

    resp = http_client.post("/api/v1/tasks/guide", data={"task": "Fold clothes"})
    assert resp.status_code == 502


def test_guide_task_empty_model_response_returns_502(http_client):
    mock = MagicMock()
    mock.generate_text = AsyncMock(
        side_effect=VisionModelEmptyResponseError("Empty model response")
    )
    override_client(mock)

    resp = http_client.post("/api/v1/tasks/guide", data={"task": "Fold clothes"})
    assert resp.status_code == 502


# ── 7. Invalid or malformed model output handled safely ──────────────────────

def test_guide_task_malformed_json_fallback(http_client):
    # Model returns non-JSON bullet list instead of JSON
    raw_output = """1. Check microwave door is closed.
2. Press start button.
3. Wait for beep."""
    mock = make_mock_client(raw_output)
    override_client(mock)

    resp = http_client.post("/api/v1/tasks/guide", data={"task": "Heat soup"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "Heat soup"
    assert len(data["steps"]) >= 1
    assert "Check microwave" in data["current_step"]


# ── 8. Regression of existing endpoints ──────────────────────────────────────

def test_regression_describe_still_works(http_client):
    mock = make_mock_client("A quiet garden.")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/describe",
        files={"image": ("garden.jpg", JPEG_IMAGE, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"description": "A quiet garden."}


def test_regression_read_text_still_works(http_client):
    mock = make_mock_client("NO SMOKING")
    override_client(mock)

    resp = http_client.post(
        "/api/v1/vision/read-text",
        files={"image": ("sign.jpg", JPEG_IMAGE, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "NO SMOKING"}
