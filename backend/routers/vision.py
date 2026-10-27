from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

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


class DescribeResponse(BaseModel):
    description: str


def _infer_mime_type(header: bytes) -> str | None:
    for signature, mime in FILE_SIGNATURES:
        if header.startswith(signature):
            return mime
    # Additional WebP check: bytes 8–11 must be b"WEBP"
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def get_vision_client() -> VisionModelClient:
    return VisionModelClient()


@router.post("/describe", response_model=DescribeResponse)
async def describe_scene(
    image: UploadFile = File(...),
    question: str | None = Form(default=None),
    client: VisionModelClient = Depends(get_vision_client),
) -> DescribeResponse:
    # Validate MIME type declared by client
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

    # Validate file signature to prevent MIME type spoofing
    inferred = _infer_mime_type(image_bytes[:12])
    if inferred is None or inferred not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content does not match a supported image format (jpeg, png, webp).",
        )

    prompt = question.strip() if (question and question.strip()) else DEFAULT_QUESTION

    try:
        description = await client.generate_vision(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=inferred,
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

    return DescribeResponse(description=description)
