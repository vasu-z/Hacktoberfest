import io
from PIL import Image, ImageEnhance, ImageOps

MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

MAX_DIMENSION = 2560


class ImageProcessingError(Exception):
    """Raised when image preprocessing fails due to corruption or limits."""


def preprocess_image_for_text_reading(
    image_bytes: bytes,
    mime_type: str,
    mode: str = "balanced",
) -> tuple[bytes, str]:
    """Preprocess image in-memory specifically for the text-reading endpoint.

    - Corrects EXIF orientation.
    - Preserves color channels and aspect ratio.
    - Avoids resizing unless dimensions exceed MAX_DIMENSION (2560px).
    - In balanced and accurate modes, applies mild contrast and sharpness enhancement.
    - In fast mode, bypasses filtering to minimize latency.
    - Keeps all data in-memory without touching disk.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Image.DecompressionBombError as e:
        raise ImageProcessingError(f"Image exceeds maximum pixel limit: {e}") from e
    except Exception as e:
        raise ImageProcessingError(f"Failed to decode image: {e}") from e

    try:
        transposed = ImageOps.exif_transpose(image)
        if transposed is not None:
            image = transposed

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        if max(image.width, image.height) > MAX_DIMENSION:
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

        if mode in ("balanced", "accurate"):
            contrast_factor = 1.10 if mode == "balanced" else 1.15
            sharpness_factor = 1.15 if mode == "balanced" else 1.20
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

        output_buffer = io.BytesIO()

        if mime_type == "image/png":
            image.save(output_buffer, format="PNG", optimize=False)
            out_mime = "image/png"
        elif mime_type == "image/webp":
            if image.mode == "RGBA":
                image.save(output_buffer, format="WEBP", quality=92)
            else:
                image.convert("RGB").save(output_buffer, format="WEBP", quality=92)
            out_mime = "image/webp"
        else:
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            elif image.mode != "RGB":
                image = image.convert("RGB")
            image.save(output_buffer, format="JPEG", quality=92)
            out_mime = "image/jpeg"

        return output_buffer.getvalue(), out_mime
    except Exception as e:
        raise ImageProcessingError(f"Image preprocessing failed: {e}") from e
