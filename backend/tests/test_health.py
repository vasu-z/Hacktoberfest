import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check_status_code():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_check_response_body():
    response = client.get("/api/v1/health")
    assert response.json() == {
        "status": "ok",
        "service": "saarthi-backend",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    test_health_check_status_code()
    test_health_check_response_body()
    print("All health endpoint tests passed successfully.")
