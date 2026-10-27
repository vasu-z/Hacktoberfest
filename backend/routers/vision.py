from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from google.genai import types
from pydantic import BaseModel

from backend.config import settings
from backend.services.image_processor import (
    ImageProcessingError,
    preprocess_image_for_text_reading,
)
from backend.services.vision_model import (
    VisionModelClient,
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])

MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 8 MB

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

# Magic bytes used to validate the file signature, regardless of client filename.
FILE_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),  # WebP files start with RIFF....WEBP
]

DEFAULT_QUESTION = (
    "Describe this scene for a blind or low-vision user. "
    "Identify clearly visible objects, their positions relative to each other, "
    "any readable text, and potential obstacles. "
    "Only describe what is clearly visible. "
    "Express uncertainty when a detail is unclear rather than guessing."
)

READ_TEXT_PROMPT = (
    "Transcribe all visible text, handwritten notes, mathematical expressions, and symbols in this image faithfully without summarizing or interpreting.\n"
    "- Preserve the original language, spelling, punctuation, capitalization, and natural reading order.\n"
    "- Faithfully transcribe mathematical symbols, equations, formulas, superscripts, subscripts, and matrix notation.\n"
    "- Never silently correct, complete, or alter the author's writing.\n"
    "- Mark illegible or uncertain words and symbols as [unclear] rather than guessing.\n"
    "- Do not invent headings, commentary, or repeating duplicate text not present in the image.\n"
    "- Provide a clear transcription suitable for reading aloud to a visually impaired user.\n"
    "- If no readable text or symbols are visible in the image, state that no readable text was detected."
)

VALID_MODES = {"fast", "balanced", "accurate"}

MODE_CONFIGS: dict[str, types.GenerateContentConfig] = {
    "fast": types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=512,
    ),
    "balanced": types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2048,
    ),
    "accurate": types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=4096,
    ),
}


class DescribeResponse(BaseModel):
    description: str


class ReadTextResponse(BaseModel):
    text: str


def _infer_mime_type(header: bytes) -> str | None:
    for signature, mime in FILE_SIGNATURES:
        if header.startswith(signature):
            return mime
    # Additional WebP check: bytes 8–11 must be b"WEBP"
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _validate_and_read_image(image: UploadFile) -> tuple[bytes, str]:
    content_type = (image.content_type or "").lower().split(";")[0].strip()
    if content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}'. Supported: jpeg, png, webp.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=422,
            detail="Image file is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the maximum allowed size of {MAX_IMAGE_SIZE // (1024 * 1024)} MB.",
        )

    inferred = _infer_mime_type(image_bytes[:12])
    if inferred is None or inferred not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content does not match a supported image format (jpeg, png, webp).",
        )

    return image_bytes, inferred


async def _generate_vision_content(
    client: VisionModelClient,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    config: types.GenerateContentConfig | None = None,
    model: str | None = None,
) -> str:
    try:
        return await client.generate_vision(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            config=config,
            model=model,
        )
    except VisionModelConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI credentials are not configured. Contact the server administrator.",
        ) from exc
    except VisionModelEmptyResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned an empty response. Please try again.",
        ) from exc
    except VisionModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model could not process the image. Please try again.",
        ) from exc


def get_vision_client() -> VisionModelClient:
    return VisionModelClient()


@router.post("/describe", response_model=DescribeResponse)
async def describe_scene(
    image: UploadFile = File(...),
    question: str | None = Form(default=None),
    client: VisionModelClient = Depends(get_vision_client),
) -> DescribeResponse:
    image_bytes, mime_type = await _validate_and_read_image(image)
    prompt = question.strip() if (question and question.strip()) else DEFAULT_QUESTION
    description = await _generate_vision_content(client, prompt, image_bytes, mime_type)
    return DescribeResponse(description=description)


@router.post("/read-text", response_model=ReadTextResponse)
async def read_text(
    image: UploadFile = File(...),
    mode: str = Form(default="balanced"),
    client: VisionModelClient = Depends(get_vision_client),
) -> ReadTextResponse:
    normalized_mode = (mode or "balanced").strip().lower()
    if normalized_mode not in VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{mode}'. Supported modes: fast, balanced, accurate.",
        )

    raw_bytes, raw_mime = await _validate_and_read_image(image)

    try:
        processed_bytes, mime_type = preprocess_image_for_text_reading(
            image_bytes=raw_bytes,
            mime_type=raw_mime,
            mode=normalized_mode,
        )
    except ImageProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Image content is corrupted or unreadable.",
        ) from exc

    gen_config = MODE_CONFIGS[normalized_mode]
    text = await _generate_vision_content(
        client=client,
        prompt=READ_TEXT_PROMPT,
        image_bytes=processed_bytes,
        mime_type=mime_type,
        config=gen_config,
        model=settings.gemini_ocr_model,
    )
    return ReadTextResponse(text=text)
