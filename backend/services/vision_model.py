from typing import Optional

from google import genai
from google.genai import types

from backend.config import settings


class VisionModelError(Exception):
    """Base exception for vision model operations."""


class VisionModelConfigurationError(VisionModelError):
    """Raised when required model credentials or configurations are missing."""


class VisionModelEmptyResponseError(VisionModelError):
    """Raised when the model returns an empty response."""


class VisionModelClient:
    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model: Optional[str] = None,
    ):
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model or settings.gemini_model

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client

        api_key = settings.gemini_api_key
        if not api_key:
            raise VisionModelConfigurationError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in the environment or .env file."
            )

        self._client = genai.Client(api_key=api_key)
        return self._client

    async def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or blank")

        client = self._get_client()
        target_model = model or self.model

        try:
            response = await client.aio.models.generate_content(
                model=target_model,
                contents=prompt.strip(),
            )
        except Exception as e:
            raise VisionModelError(f"Model generation request failed: {type(e).__name__}") from e

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise VisionModelEmptyResponseError("Model returned an empty or blank response")

        return text.strip()

    async def generate_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        config: Optional[types.GenerateContentConfig] = None,
        model: Optional[str] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or blank")
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")
        if not mime_type or not mime_type.strip():
            raise ValueError("mime_type must be specified")

        client = self._get_client()

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type.strip())
        contents = [prompt.strip(), image_part]

        kwargs = {}
        if config is not None:
            kwargs["config"] = config

        target_model = model or self.model

        try:
            response = await client.aio.models.generate_content(
                model=target_model,
                contents=contents,
                **kwargs,
            )
        except Exception as e:
            raise VisionModelError(f"Model vision generation request failed: {type(e).__name__}") from e

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise VisionModelEmptyResponseError("Model returned an empty or blank response")

        return text.strip()
