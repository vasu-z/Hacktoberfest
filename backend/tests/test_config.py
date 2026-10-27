import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from backend.config import Settings
from backend.main import app

client = TestClient(app)


def test_default_settings():
    settings = Settings(_env_file=None)
    assert settings.saarthi_service_name == "saarthi-backend"
    assert settings.saarthi_version == "0.1.0"
    assert settings.saarthi_env == "development"
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemma-4-26b-a4b-it"
    assert settings.gemini_ocr_model == "gemini-3.5-flash-lite"


def test_env_variable_overrides():
    env_overrides = {
        "SAARTHI_SERVICE_NAME": "saarthi-custom",
        "SAARTHI_VERSION": "1.2.3",
        "SAARTHI_ENV": "staging",
        "GEMINI_API_KEY": "test-key-override",
        "GEMINI_MODEL": "custom-vision-model",
        "GEMINI_OCR_MODEL": "custom-ocr-model",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        settings = Settings(_env_file=None)
        assert settings.saarthi_service_name == "saarthi-custom"
        assert settings.saarthi_version == "1.2.3"
        assert settings.saarthi_env == "staging"
        assert settings.gemini_api_key == "test-key-override"
        assert settings.gemini_model == "custom-vision-model"
        assert settings.gemini_ocr_model == "custom-ocr-model"


def test_optional_api_key_and_secret_masking():
    settings_no_key = Settings(_env_file=None, gemini_api_key=None)
    assert settings_no_key.gemini_api_key is None

    settings_with_key = Settings(_env_file=None, gemini_api_key="secret-key-sample")
    assert settings_with_key.gemini_api_key == "secret-key-sample"
    assert "secret-key-sample" not in repr(settings_with_key)
    assert "**********" in repr(settings_with_key)


def test_configured_model_name():
    settings = Settings(_env_file=None, gemini_model="gemma-4-26b-a4b-it")
    assert settings.gemini_model == "gemma-4-26b-a4b-it"

    custom_model_settings = Settings(_env_file=None, gemini_model="custom-gemini-model")
    assert custom_model_settings.gemini_model == "custom-gemini-model"


def test_application_starts_and_health_behavior_unchanged():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "saarthi-backend",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    test_default_settings()
    test_env_variable_overrides()
    test_optional_api_key_and_secret_masking()
    test_configured_model_name()
    test_application_starts_and_health_behavior_unchanged()
    print("All configuration tests passed successfully.")
