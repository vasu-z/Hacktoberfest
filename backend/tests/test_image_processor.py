import io
from PIL import Image
import pytest

from backend.services.image_processor import (
    MAX_DIMENSION,
    ImageProcessingError,
    preprocess_image_for_text_reading,
)


def _create_test_image(
    format_name: str,
    size: tuple[int, int] = (100, 100),
    color: tuple[int, int, int] = (255, 0, 0),
    exif_orientation: int | None = None,
) -> tuple[bytes, str]:
    img = Image.new("RGB", size, color=color)
    mime_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    mime = mime_map.get(format_name.upper(), "image/jpeg")

    buf = io.BytesIO()
    if exif_orientation is not None and format_name.upper() == "JPEG":
        exif = img.getexif()
        exif[0x0112] = exif_orientation
        img.save(buf, format=format_name, exif=exif)
    else:
        img.save(buf, format=format_name)
    return buf.getvalue(), mime


def test_preprocess_valid_formats():
    for fmt in ["JPEG", "PNG", "WEBP"]:
        raw_bytes, mime = _create_test_image(fmt, (80, 80))
        out_bytes, out_mime = preprocess_image_for_text_reading(raw_bytes, mime)
        assert len(out_bytes) > 0
        assert out_mime == mime
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (80, 80)


def test_preprocess_preserves_colors():
    # Pure red annotation
    raw_bytes, mime = _create_test_image("PNG", (40, 40), color=(220, 20, 20))
    out_bytes, _ = preprocess_image_for_text_reading(raw_bytes, mime, mode="balanced")
    out_img = Image.open(io.BytesIO(out_bytes))
    pixel = out_img.getpixel((20, 20))
    assert pixel[0] > 180
    assert pixel[1] < 60
    assert pixel[2] < 60


def test_preprocess_exif_orientation_correction():
    # 200 wide x 100 high with orientation 6 (90 degrees CW) -> should become 100 wide x 200 high
    raw_bytes, mime = _create_test_image("JPEG", (200, 100), exif_orientation=6)
    out_bytes, _ = preprocess_image_for_text_reading(raw_bytes, mime)
    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.size == (100, 200)


def test_preprocess_does_not_resize_small_dense_images():
    raw_bytes, mime = _create_test_image("PNG", (1200, 800))
    out_bytes, _ = preprocess_image_for_text_reading(raw_bytes, mime)
    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.size == (1200, 800)


def test_preprocess_caps_excessive_dimensions_preserving_aspect_ratio():
    # 4000 x 2000 has 2:1 aspect ratio
    raw_bytes, mime = _create_test_image("JPEG", (4000, 2000))
    out_bytes, _ = preprocess_image_for_text_reading(raw_bytes, mime)
    out_img = Image.open(io.BytesIO(out_bytes))
    assert max(out_img.width, out_img.height) == MAX_DIMENSION
    assert out_img.size == (2560, 1280)


def test_preprocess_invalid_corrupted_image_raises_error():
    garbage = b"\xff\xd8\xff" + b"\x00" * 30
    with pytest.raises(ImageProcessingError):
        preprocess_image_for_text_reading(garbage, "image/jpeg")


def test_preprocess_decompression_bomb_handling(monkeypatch):
    import backend.services.image_processor as ip
    # Temporarily set MAX_IMAGE_PIXELS very low to trigger decompression bomb protection
    monkeypatch.setattr(ip.Image, "MAX_IMAGE_PIXELS", 500)
    raw_bytes, mime = _create_test_image("PNG", (100, 100))  # 10,000 pixels > 500
    with pytest.raises(ImageProcessingError) as exc_info:
        ip.preprocess_image_for_text_reading(raw_bytes, mime)
    assert "pixel limit" in str(exc_info.value).lower()
