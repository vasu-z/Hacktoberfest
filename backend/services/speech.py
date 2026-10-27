"""Text-to-Speech (TTS) service using Google Gen AI Interactions API.

PRIVACY & DATA HANDLING NOTICE:
- This service sends ONLY the text transcript intended to be spoken aloud.
- Images, personal context, and raw camera feeds must NEVER be forwarded to the TTS model.
- Per Google Gemini free-tier terms of service, submitted content may be logged
  or processed by human reviewers to train and improve Google products.
- Free-tier processing is NOT private and must not be used for sensitive
  confidential personal records or medical information.
"""

import base64
from typing import Optional

from google import genai

from backend.config import settings

DEFAULT_VOICE = "Kore"
DEFAULT_VOICE_STYLE = "warm, calm, clear, friendly"
MAX_TEXT_LENGTH = 4000


class SpeechServiceError(Exception):
    """Base exception for speech synthesis operations."""


class SpeechConfigurationError(SpeechServiceError):
    """Raised when required model credentials or configurations are missing."""


class SpeechEmptyResponseError(SpeechServiceError):
    """Raised when the TTS model returns an empty or corrupt audio response."""


class SpeechRateLimitError(SpeechServiceError):
    """Raised when API rate limits are exceeded (429)."""


class SpeechTimeoutError(SpeechServiceError):
    """Raised when the TTS request times out."""


class SpeechService:
    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        self._client = client
        self._model = model
        self._voice = voice

    @property
    def model(self) -> str:
        return self._model or settings.gemini_tts_model

    @property
    def voice(self) -> str:
        return self._voice or DEFAULT_VOICE

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client

        api_key = settings.gemini_api_key
        if not api_key:
            raise SpeechConfigurationError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in the environment or .env file."
            )

        self._client = genai.Client(api_key=api_key)
        return self._client

    async def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[bytes, str]:
        clean_text = text.strip() if text else ""
        if not clean_text:
            raise ValueError("Text cannot be empty or blank.")
        if len(clean_text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds maximum supported length of {MAX_TEXT_LENGTH} characters.")

        client = self._get_client()
        target_model = model or self.model
        target_voice = voice or self.voice

        try:
            interaction = await client.aio.interactions.create(
                model=target_model,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": clean_text,
                                "annotations": [
                                    {
                                        "type": "speech_metadata",
                                        "style": DEFAULT_VOICE_STYLE,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [
                        {"voice": target_voice},
                    ]
                },
            )
        except TimeoutError as e:
            raise SpeechTimeoutError(f"TTS generation timed out: {e}") from e
        except Exception as e:
            err_str = str(e).lower()
            err_name = type(e).__name__
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                raise SpeechRateLimitError(f"TTS rate limit exceeded: {err_name}") from e
            if "timeout" in err_str or "timed out" in err_str:
                raise SpeechTimeoutError(f"TTS request timed out: {err_name}") from e
            raise SpeechServiceError(f"TTS generation request failed: {err_name}") from e

        # Extract audio payload and mime type
        audio_bytes: Optional[bytes] = None
        mime_type = "audio/wav"

        if hasattr(interaction, "output_audio") and interaction.output_audio:
            out_audio = interaction.output_audio
            if hasattr(out_audio, "data") and out_audio.data:
                raw_data = out_audio.data
                if isinstance(raw_data, str):
                    try:
                        audio_bytes = base64.b64decode(raw_data)
                    except Exception as e:
                        raise SpeechEmptyResponseError("Failed to decode base64 audio data.") from e
                elif isinstance(raw_data, bytes):
                    audio_bytes = raw_data
            if hasattr(out_audio, "mime_type") and out_audio.mime_type:
                mime_type = str(out_audio.mime_type)

        # Fallback check in steps if output_audio at top-level is absent
        if not audio_bytes and hasattr(interaction, "steps") and interaction.steps:
            for step in interaction.steps:
                if hasattr(step, "output_audio") and step.output_audio:
                    out_audio = step.output_audio
                    if hasattr(out_audio, "data") and out_audio.data:
                        raw_data = out_audio.data
                        if isinstance(raw_data, str):
                            try:
                                audio_bytes = base64.b64decode(raw_data)
                            except Exception as e:
                                raise SpeechEmptyResponseError("Failed to decode base64 audio data.") from e
                        elif isinstance(raw_data, bytes):
                            audio_bytes = raw_data
                        if hasattr(out_audio, "mime_type") and out_audio.mime_type:
                            mime_type = str(out_audio.mime_type)
                        break

        if not audio_bytes:
            raise SpeechEmptyResponseError("The TTS model returned an empty audio response.")

        return audio_bytes, mime_type
