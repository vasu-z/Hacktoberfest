import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app
from backend.routers.memory import get_memory_service
from backend.services.spatial_memory import SpatialMemoryService


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "test_spatial_memory.db"


@pytest.fixture
def temp_service(temp_db_path):
    service = SpatialMemoryService(db_path=temp_db_path)
    app.dependency_overrides[get_memory_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


@pytest.fixture
def http_client(temp_service):
    return TestClient(app)


# ── 1. Creating and retrieving a memory ──────────────────────────────────────

def test_create_and_retrieve_memory(http_client):
    create_resp = http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "user_123",
            "name": "Front Door Keys",
            "description": "On the small wooden tray beside the coat rack.",
            "location_label": "hallway entrance",
        },
    )
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    assert created_data["user_id"] == "user_123"
    assert created_data["name"] == "Front Door Keys"
    assert created_data["description"] == "On the small wooden tray beside the coat rack."
    assert created_data["location_label"] == "hallway entrance"
    assert "id" in created_data
    assert "created_at" in created_data
    assert "updated_at" in created_data

    memory_id = created_data["id"]

    get_resp = http_client.get(f"/api/v1/memory/{memory_id}?user_id=user_123")
    assert get_resp.status_code == 200
    retrieved_data = get_resp.json()
    assert retrieved_data["id"] == memory_id
    assert retrieved_data["name"] == "Front Door Keys"


# ── 2. Listing memories by user identifier ───────────────────────────────────

def test_list_memories_by_user(http_client):
    http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "alice",
            "name": "Glasses",
            "description": "On bedside table",
        },
    )
    http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "alice",
            "name": "Mug",
            "description": "On kitchen counter",
        },
    )
    http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "bob",
            "name": "Backpack",
            "description": "Near the front door",
        },
    )

    resp_alice = http_client.get("/api/v1/memory?user_id=alice")
    assert resp_alice.status_code == 200
    alice_memories = resp_alice.json()
    assert len(alice_memories) == 2
    names = {m["name"] for m in alice_memories}
    assert names == {"Glasses", "Mug"}

    resp_bob = http_client.get("/api/v1/memory?user_id=bob")
    assert resp_bob.status_code == 200
    bob_memories = resp_bob.json()
    assert len(bob_memories) == 1
    assert bob_memories[0]["name"] == "Backpack"


# ── 3. Updating and deleting memories ────────────────────────────────────────

def test_update_and_delete_memory(http_client):
    create_resp = http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "user_update",
            "name": "Headphones",
            "description": "On the desk",
            "location_label": "office",
        },
    )
    memory_id = create_resp.json()["id"]

    patch_resp = http_client.patch(
        f"/api/v1/memory/{memory_id}",
        json={
            "user_id": "user_update",
            "description": "Hanging on the desk hook",
            "location_label": "office desk hook",
        },
    )
    assert patch_resp.status_code == 200
    patched_data = patch_resp.json()
    assert patched_data["name"] == "Headphones"
    assert patched_data["description"] == "Hanging on the desk hook"
    assert patched_data["location_label"] == "office desk hook"
    assert patched_data["updated_at"] >= patched_data["created_at"]

    del_resp = http_client.delete(f"/api/v1/memory/{memory_id}?user_id=user_update")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    get_resp = http_client.get(f"/api/v1/memory/{memory_id}?user_id=user_update")
    assert get_resp.status_code == 404


# ── 4. Empty or invalid input ────────────────────────────────────────────────

def test_create_memory_empty_fields_returns_422(http_client):
    # Blank user_id
    resp = http_client.post(
        "/api/v1/memory",
        json={"user_id": "   ", "name": "Item", "description": "Desc"},
    )
    assert resp.status_code == 422

    # Blank name
    resp = http_client.post(
        "/api/v1/memory",
        json={"user_id": "user1", "name": "   ", "description": "Desc"},
    )
    assert resp.status_code == 422

    # Blank description
    resp = http_client.post(
        "/api/v1/memory",
        json={"user_id": "user1", "name": "Item", "description": "   "},
    )
    assert resp.status_code == 422


def test_list_and_get_memory_missing_user_id_returns_422(http_client):
    resp_list = http_client.get("/api/v1/memory")
    assert resp_list.status_code == 422

    resp_get = http_client.get("/api/v1/memory/some-id")
    assert resp_get.status_code == 422


def test_patch_memory_without_update_fields_returns_422(http_client):
    resp = http_client.patch(
        "/api/v1/memory/some-id",
        json={"user_id": "user1"},
    )
    assert resp.status_code == 422


# ── 5. Missing records return 404 ─────────────────────────────────────────────

def test_missing_records_return_404(http_client):
    fake_id = "non-existent-uuid"
    assert http_client.get(f"/api/v1/memory/{fake_id}?user_id=user1").status_code == 404
    assert (
        http_client.patch(
            f"/api/v1/memory/{fake_id}",
            json={"user_id": "user1", "name": "New Name"},
        ).status_code
        == 404
    )
    assert http_client.delete(f"/api/v1/memory/{fake_id}?user_id=user1").status_code == 404


# ── 6. Cross-user record access isolation ────────────────────────────────────

def test_cross_user_isolation(http_client):
    create_resp = http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "user_private",
            "name": "Secret Diary",
            "description": "Under mattress",
        },
    )
    memory_id = create_resp.json()["id"]

    # Intruder tries to read user_private's record
    get_resp = http_client.get(f"/api/v1/memory/{memory_id}?user_id=intruder")
    assert get_resp.status_code == 404

    # Intruder tries to patch user_private's record
    patch_resp = http_client.patch(
        f"/api/v1/memory/{memory_id}",
        json={"user_id": "intruder", "name": "Compromised"},
    )
    assert patch_resp.status_code == 404

    # Intruder tries to delete user_private's record
    del_resp = http_client.delete(f"/api/v1/memory/{memory_id}?user_id=intruder")
    assert del_resp.status_code == 404

    # Original owner can still read intact memory
    owner_resp = http_client.get(f"/api/v1/memory/{memory_id}?user_id=user_private")
    assert owner_resp.status_code == 200
    assert owner_resp.json()["name"] == "Secret Diary"


# ── 7. Persistence across separate connections / instances ───────────────────

def test_persistence_across_separate_connections(temp_db_path):
    # Instance 1 writes
    service_1 = SpatialMemoryService(db_path=temp_db_path)
    created = service_1.create_memory(
        user_id="persist_user",
        name="Wallet",
        description="In jacket pocket",
        location_label="hallway closet",
    )
    memory_id = created.id

    # Instance 2 reads from same file
    service_2 = SpatialMemoryService(db_path=temp_db_path)
    loaded = service_2.get_memory(memory_id=memory_id, user_id="persist_user")
    assert loaded is not None
    assert loaded.id == memory_id
    assert loaded.name == "Wallet"
    assert loaded.description == "In jacket pocket"
    assert loaded.location_label == "hallway closet"


# ── 8. Correct handling of optional location labels ──────────────────────────

def test_optional_location_label_handling(http_client):
    # 1. Create with location_label
    resp_with = http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "u_label",
            "name": "Lamp",
            "description": "Reading lamp",
            "location_label": "corner table",
        },
    )
    assert resp_with.status_code == 201
    assert resp_with.json()["location_label"] == "corner table"
    mem_id = resp_with.json()["id"]

    # 2. Clear location_label explicitly via null or empty
    patch_clear = http_client.patch(
        f"/api/v1/memory/{mem_id}",
        json={
            "user_id": "u_label",
            "location_label": None,
        },
    )
    assert patch_clear.status_code == 200
    assert patch_clear.json()["location_label"] is None

    # Verify retrieval reflects cleared label
    get_cleared = http_client.get(f"/api/v1/memory/{mem_id}?user_id=u_label")
    assert get_cleared.json()["location_label"] is None

    # 3. Create without location_label
    resp_without = http_client.post(
        "/api/v1/memory",
        json={
            "user_id": "u_label",
            "name": "Notebook",
            "description": "Spiral notebook",
        },
    )
    assert resp_without.status_code == 201
    assert resp_without.json()["location_label"] is None
