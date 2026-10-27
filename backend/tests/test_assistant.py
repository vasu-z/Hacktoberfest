import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app
from backend.routers.memory import get_memory_service
from backend.routers.vision import get_vision_client
from backend.services.spatial_memory import SpatialMemoryService
from backend.services.vision_model import (
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "test_assistant_memory.db"


@pytest.fixture
def temp_memory_service(temp_db_path):
    service = SpatialMemoryService(db_path=temp_db_path)
    app.dependency_overrides[get_memory_service] = lambda: service
    return service


@pytest.fixture
def mock_vision_client():
    mock = MagicMock()
    mock.generate_text = AsyncMock(return_value="General assistant spoken response.")
    mock.generate_vision = AsyncMock(return_value="Visual response.")
    app.dependency_overrides[get_vision_client] = lambda: mock
    return mock


@pytest.fixture
def http_client(temp_memory_service, mock_vision_client):
    return TestClient(app)


# ── 1. Blank and invalid requests are rejected ──────────────────────────────

def test_ask_blank_message_returns_422(http_client):
    resp = http_client.post("/api/v1/assistant/ask", json={"message": "   "})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_ask_empty_message_returns_422(http_client):
    resp = http_client.post("/api/v1/assistant/ask", json={"message": ""})
    assert resp.status_code == 422


def test_ask_missing_message_returns_422(http_client):
    resp = http_client.post("/api/v1/assistant/ask", json={"intent": "general"})
    assert resp.status_code == 422


def test_ask_invalid_intent_returns_422(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "What is the time?", "intent": "superpower"},
    )
    assert resp.status_code == 422
    assert "Invalid intent" in resp.json()["detail"]


# ── 2. Obvious messages resolve to the appropriate intent (auto routing) ────

def test_auto_route_read_text(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Can you read this note for me?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "read_text"
    assert data["action_required"] is True
    assert data["required_input"] == "image"
    assert "read" in data["response"].lower() or "image" in data["response"].lower()


def test_auto_route_scene(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Describe what is in front of me."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "scene"
    assert data["action_required"] is True
    assert data["required_input"] == "image"
    assert "camera" in data["response"].lower() or "image" in data["response"].lower()


def test_auto_route_object(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Find my keys on the desk."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "object"
    assert data["action_required"] is True
    assert data["required_input"] == "image"
    assert "keys" in data["response"].lower()


def test_auto_route_task(http_client, mock_vision_client):
    mock_payload = {
        "task": "Make tea",
        "steps": ["Boil water", "Pour into mug"],
        "current_step": "Boil fresh water in the kettle.",
        "needs_visual_confirmation": False,
        "cautions": ["Beware of hot water."],
    }
    mock_vision_client.generate_text.return_value = json.dumps(mock_payload)

    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "How do I make tea?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "task"
    assert data["action_required"] is False
    assert "Boil fresh water" in data["response"]


# ── 3. Ambiguous messages produce a clarifying response ─────────────────────

@pytest.mark.parametrize("msg", ["hello", "help", "can you help me", "check"])
def test_ambiguous_message_produces_clarification(http_client, msg):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": msg},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "clarification"
    assert data["action_required"] is True
    assert "help" in data["response"].lower() or "what would you like" in data["response"].lower()


# ── 4. Scene, text, and object identify missing image input ─────────────────

def test_explicit_scene_requires_image(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Look around", "intent": "scene"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "scene"
    assert data["action_required"] is True
    assert data["required_input"] == "image"


def test_explicit_read_text_requires_image(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Read this", "intent": "read_text"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "read_text"
    assert data["action_required"] is True
    assert data["required_input"] == "image"


def test_explicit_object_without_target_requests_both(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Find it", "intent": "object"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "object"
    assert data["action_required"] is True
    assert "target_object" in data["required_input"]


# ── 5. Task and memory reuse existing functionality without false success ───

def test_task_guidance_dispatch_with_visual_confirmation(http_client, mock_vision_client):
    mock_payload = {
        "task": "Assemble desk fan",
        "steps": ["Attach fan blade", "Screw the front grill"],
        "current_step": "Attach the fan blade firmly onto the motor shaft.",
        "needs_visual_confirmation": True,
        "cautions": ["Ensure power is unplugged."],
    }
    mock_vision_client.generate_text.return_value = json.dumps(mock_payload)

    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={
            "message": "Guide me to assemble this desk fan",
            "intent": "task",
            "context": "Desk fan in pieces",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "task"
    assert data["action_required"] is True
    assert data["required_input"] == "image for visual confirmation"
    assert "Attach the fan blade" in data["response"]
    assert "Ensure power is unplugged." in data["cautions"]


def test_memory_missing_user_id_requires_input(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Where did I put my keys?", "intent": "memory"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "memory"
    assert data["action_required"] is True
    assert data["required_input"] == "user_id"
    assert "user identifier" in data["response"]


def test_memory_save_and_recall_real_database(http_client, temp_memory_service):
    # Save a memory item
    save_resp = http_client.post(
        "/api/v1/assistant/ask",
        json={
            "message": "Remember that my passport is on the top shelf of the closet",
            "intent": "memory",
            "user_id": "traveler_1",
        },
    )
    assert save_resp.status_code == 200
    save_data = save_resp.json()
    assert save_data["intent"] == "memory"
    assert save_data["action_required"] is False
    assert "passport" in save_data["response"].lower()

    # Verify directly in SQLite service
    memories = temp_memory_service.list_memories(user_id="traveler_1")
    assert len(memories) == 1
    assert memories[0].name.lower() == "passport"

    # Recall the memory item
    recall_resp = http_client.post(
        "/api/v1/assistant/ask",
        json={
            "message": "Where did I put my passport?",
            "intent": "auto",
            "user_id": "traveler_1",
        },
    )
    assert recall_resp.status_code == 200
    recall_data = recall_resp.json()
    assert recall_data["intent"] == "memory"
    assert recall_data["action_required"] is False
    assert "top shelf" in recall_data["response"].lower()


def test_memory_recall_not_found(http_client):
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={
            "message": "Where did I leave my glasses?",
            "intent": "memory",
            "user_id": "empty_user",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "memory"
    assert "could not find" in data["response"].lower()


def test_memory_list_empty_and_populated(http_client, temp_memory_service):
    # Empty
    resp_empty = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "List my memories", "user_id": "user_a"},
    )
    assert resp_empty.status_code == 200
    assert "do not have any saved" in resp_empty.json()["response"].lower()

    # Add items
    temp_memory_service.create_memory(
        user_id="user_a",
        name="House Keys",
        description="Beside the door",
    )
    resp_list = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Show my memories", "user_id": "user_a"},
    )
    assert resp_list.status_code == 200
    assert "House Keys" in resp_list.json()["response"]


# ── 6. General requests use AI client and handle errors ─────────────────────

def test_general_request_success(http_client, mock_vision_client):
    mock_vision_client.generate_text.return_value = "Paris is the capital of France."
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "What is the capital of France?", "intent": "general"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "general"
    assert data["response"] == "Paris is the capital of France."
    assert data["action_required"] is False
    mock_vision_client.generate_text.assert_awaited_once()


def test_general_missing_credentials_returns_503(http_client, mock_vision_client):
    mock_vision_client.generate_text.side_effect = VisionModelConfigurationError("Missing key")
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Tell me a story", "intent": "general"},
    )
    assert resp.status_code == 503
    assert "AI credentials are not configured" in resp.json()["detail"]


def test_general_empty_model_response_returns_502(http_client, mock_vision_client):
    mock_vision_client.generate_text.side_effect = VisionModelEmptyResponseError("Empty")
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Explain quantum physics", "intent": "general"},
    )
    assert resp.status_code == 502
    assert "empty response" in resp.json()["detail"]


def test_general_provider_error_returns_502(http_client, mock_vision_client):
    mock_vision_client.generate_text.side_effect = VisionModelError("API down")
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "Hello general", "intent": "general"},
    )
    assert resp.status_code == 502
    assert "could not process" in resp.json()["detail"]


# ── 7. Speech-ready response checks ─────────────────────────────────────────

def test_response_is_speech_friendly_and_valid_schema(http_client, mock_vision_client):
    mock_vision_client.generate_text.return_value = "The weather today is mild with partly cloudy skies."
    resp = http_client.post(
        "/api/v1/assistant/ask",
        json={"message": "How is the weather?", "intent": "general"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["intent"], str)
    assert isinstance(data["response"], str)
    assert isinstance(data["action_required"], bool)
    assert isinstance(data["cautions"], list)

    # Check that speech output does not contain markdown tables, headings, or bullets
    assert "#" not in data["response"]
    assert "|" not in data["response"]
    assert "*" not in data["response"]
