import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import Settings
from backend.services.vision_model import (
    VisionModelClient,
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)


@pytest.fixture
def mock_sdk_client():
    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    return mock_client


@pytest.mark.anyio
async def test_text_generation_success(mock_sdk_client):
    mock_response = MagicMock()
    mock_response.text = "A brightly lit kitchen with a wooden table."
    mock_sdk_client.aio.models.generate_content.return_value = mock_response

    client = VisionModelClient(client=mock_sdk_client, model="gemma-4-26b-a4b-it")
    result = await client.generate_text("Describe what is in front of me.")

    assert result == "A brightly lit kitchen with a wooden table."
    mock_sdk_client.aio.models.generate_content.assert_awaited_once_with(
        model="gemma-4-26b-a4b-it",
        contents="Describe what is in front of me.",
    )


@pytest.mark.anyio
async def test_image_generation_success(mock_sdk_client):
    mock_response = MagicMock()
    mock_response.text = "There is a blue coffee mug on the counter."
    mock_sdk_client.aio.models.generate_content.return_value = mock_response

    client = VisionModelClient(client=mock_sdk_client, model="gemma-4-26b-a4b-it")
    image_bytes = b"fake_jpeg_binary_data"
    mime_type = "image/jpeg"

    result = await client.generate_vision(
        prompt="Identify the object.",
        image_bytes=image_bytes,
        mime_type=mime_type,
    )

    assert result == "There is a blue coffee mug on the counter."
    mock_sdk_client.aio.models.generate_content.assert_awaited_once()
    call_args = mock_sdk_client.aio.models.generate_content.await_args
    assert call_args.kwargs["model"] == "gemma-4-26b-a4b-it"
    contents = call_args.kwargs["contents"]
    assert contents[0] == "Identify the object."
    assert isinstance(contents[1], types.Part)


@pytest.mark.anyio
async def test_generate_vision_model_override(mock_sdk_client):
    mock_response = MagicMock()
    mock_response.text = "Model text."
    mock_sdk_client.aio.models.generate_content.return_value = mock_response

    client = VisionModelClient(client=mock_sdk_client, model="gemma-4-26b-a4b-it")
    await client.generate_vision(
        prompt="Read text",
        image_bytes=b"fake_jpeg",
        mime_type="image/jpeg",
        model="gemini-3.5-flash-lite",
    )
    call_args = mock_sdk_client.aio.models.generate_content.await_args
    assert call_args.kwargs["model"] == "gemini-3.5-flash-lite"


@pytest.mark.anyio
async def test_missing_api_key_raises_configuration_error():
    with patch("backend.services.vision_model.settings", Settings(_env_file=None, gemini_api_key=None)):
        client = VisionModelClient()
        with pytest.raises(VisionModelConfigurationError) as exc_info:
            await client.generate_text("Hello")
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.anyio
async def test_provider_exception_translated_to_service_error(mock_sdk_client):
    mock_sdk_client.aio.models.generate_content.side_effect = RuntimeError("API service unavailable")

    client = VisionModelClient(client=mock_sdk_client)
    with pytest.raises(VisionModelError) as exc_info:
        await client.generate_text("Describe scene")

    assert "Model generation request failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.anyio
async def test_empty_model_response_raises_error(mock_sdk_client):
    mock_response = MagicMock()
    mock_response.text = "   "
    mock_sdk_client.aio.models.generate_content.return_value = mock_response

    client = VisionModelClient(client=mock_sdk_client)
    with pytest.raises(VisionModelEmptyResponseError):
        await client.generate_text("Describe scene")


@pytest.mark.anyio
async def test_invalid_input_rejected_before_api_call(mock_sdk_client):
    client = VisionModelClient(client=mock_sdk_client)

    with pytest.raises(ValueError, match="Prompt cannot be empty or blank"):
        await client.generate_text("   ")

    with pytest.raises(ValueError, match="Prompt cannot be empty or blank"):
        await client.generate_vision("", b"data", "image/jpeg")

    with pytest.raises(ValueError, match="image_bytes cannot be empty"):
        await client.generate_vision("prompt", b"", "image/jpeg")

    with pytest.raises(ValueError, match="mime_type must be specified"):
        await client.generate_vision("prompt", b"data", "")

    mock_sdk_client.aio.models.generate_content.assert_not_awaited()
