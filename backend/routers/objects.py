import json
import re
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.genai import types
from pydantic import BaseModel

from backend.routers.vision import (
    _generate_vision_content,
    _validate_and_read_image,
    get_vision_client,
)
from backend.services.vision_model import VisionModelClient

router = APIRouter(prefix="/api/v1/objects", tags=["objects"])

CertaintyLevel = Literal["clear", "uncertain", "not_visible"]

OBJECT_ASSIST_PROMPT_TEMPLATE = (
    "You are an AI vision companion assisting a blind or low-vision user.\n"
    "Look at this image and determine whether the requested target object is visible: \"{target_object}\".\n\n"
    "Guidelines:\n"
    "1. If the target object is clearly visible:\n"
    "   - Set found to true.\n"
    "   - Set certainty to \"clear\".\n"
    "   - In description, concisely describe what it looks like and its approximate position relative to visible landmarks (e.g. \"on the desk to the left of the laptop\").\n"
    "2. If an object resembles the target but is partially obscured, blurry, or ambiguous:\n"
    "   - Set found to true if likely present, or false if too ambiguous.\n"
    "   - Set certainty to \"uncertain\".\n"
    "   - In description, describe what is visible and clearly express uncertainty.\n"
    "3. If the target object is not visible:\n"
    "   - Set found to false.\n"
    "   - Set certainty to \"not_visible\".\n"
    "   - In description, state that the object cannot be identified in the image.\n"
    "4. Safety rules:\n"
    "   - Never invent locations, precise coordinates, or distances.\n"
    "   - Do not claim that the path to the object is safe or obstacle-free.\n"
    "   - Provide practical visual assistance, not navigation guarantees.\n\n"
    "Return ONLY a JSON object with this exact schema:\n"
    "{{\n"
    '  "found": true,\n'
    '  "description": "<concise description and relative position>",\n'
    '  "certainty": "clear"\n'
    "}}"
)

OBJECT_ASSIST_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.1,
    max_output_tokens=1024,
)


class ObjectAssistResponse(BaseModel):
    target_object: str
    found: bool
    description: str
    certainty: CertaintyLevel


def _clean_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_and_validate_model_response(raw_text: str, target_object: str) -> ObjectAssistResponse:
    cleaned = _clean_json_markdown(raw_text)
    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                pass

    if isinstance(data, dict):
        found = bool(data.get("found", False))
        certainty_raw = str(data.get("certainty", "")).lower().strip()
        if certainty_raw in ("clear", "uncertain", "not_visible"):
            certainty = certainty_raw
        else:
            certainty = "clear" if found else "not_visible"

        description = str(data.get("description", "")).strip()
        if not description:
            description = (
                f"The {target_object} is visible in the scene."
                if found
                else f"The {target_object} was not identified in the image."
            )

        return ObjectAssistResponse(
            target_object=target_object,
            found=found,
            description=description,
            certainty=certainty,
        )

    # Fallback if model output is unstructured plain text
    text_lower = raw_text.lower()
    if any(neg in text_lower for neg in ["not visible", "not found", "cannot see", "cannot find", "no "]):
        return ObjectAssistResponse(
            target_object=target_object,
            found=False,
            description=raw_text.strip() or f"The {target_object} was not identified in the image.",
            certainty="not_visible",
        )

    return ObjectAssistResponse(
        target_object=target_object,
        found=True,
        description=raw_text.strip(),
        certainty="uncertain",
    )


@router.post("/assist", response_model=ObjectAssistResponse)
async def assist_object(
    image: UploadFile = File(...),
    target_object: str = Form(...),
    client: VisionModelClient = Depends(get_vision_client),
) -> ObjectAssistResponse:
    clean_target = target_object.strip() if target_object else ""
    if not clean_target:
        raise HTTPException(
            status_code=422,
            detail="target_object must not be empty or blank.",
        )

    image_bytes, mime_type = await _validate_and_read_image(image)

    prompt = OBJECT_ASSIST_PROMPT_TEMPLATE.format(target_object=clean_target)

    raw_response = await _generate_vision_content(
        client=client,
        prompt=prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
        config=OBJECT_ASSIST_CONFIG,
    )

    return _parse_and_validate_model_response(
        raw_text=raw_response,
        target_object=clean_target,
    )
