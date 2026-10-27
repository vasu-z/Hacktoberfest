import json
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from google.genai import types
from pydantic import BaseModel, Field

from backend.routers.vision import _validate_and_read_image, get_vision_client
from backend.services.vision_model import (
    VisionModelClient,
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

TASK_GUIDANCE_SYSTEM_PROMPT = """You are SAARTHI, an AI vision companion providing step-by-step task guidance for a blind or low-vision user.

GUIDELINES & SAFETY CONSTRAINTS:
1. Provide concise, clear, actionable steps suitable for reading aloud.
2. Identify the single immediate next action as "current_step".
3. If an image is provided, use it strictly as visual context. Never invent visual details.
4. Set "needs_visual_confirmation" to true if a follow-up image would help verify step completion or orientation; otherwise false.
5. In "cautions", clearly communicate relevant uncertainties, physical obstacles, or warnings.
6. Safety rules:
   - Never claim an action was completed without direct visual evidence.
   - Never claim a walking path, surface, or environment is safe or hazard-free based solely on an image.
   - Refuse dangerous instructions involving prescription medication dosages, hazardous power tools, or high-risk actions.
7. Return ONLY valid JSON matching this schema:
{
  "task": "<the user task>",
  "steps": ["<step 1>", "<step 2>", ...],
  "current_step": "<immediate next step>",
  "needs_visual_confirmation": true | false,
  "cautions": ["<caution 1>", ...]
}
"""

TASK_GUIDANCE_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.2,
    max_output_tokens=1536,
)


class TaskGuidanceResponse(BaseModel):
    task: str
    steps: list[str] = Field(default_factory=list)
    current_step: str
    needs_visual_confirmation: bool = False
    cautions: list[str] = Field(default_factory=list)


def _clean_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_and_validate_guidance(raw_text: str, task: str) -> TaskGuidanceResponse:
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
        steps = data.get("steps")
        if isinstance(steps, list):
            clean_steps = [str(s).strip() for s in steps if str(s).strip()]
        else:
            clean_steps = []

        current_step = str(data.get("current_step", "")).strip()
        if not current_step and clean_steps:
            current_step = clean_steps[0]
        elif not current_step:
            current_step = "Proceed with initial preparation."

        cautions = data.get("cautions")
        if isinstance(cautions, list):
            clean_cautions = [str(c).strip() for c in cautions if str(c).strip()]
        else:
            clean_cautions = []

        needs_visual = bool(data.get("needs_visual_confirmation", False))

        return TaskGuidanceResponse(
            task=str(data.get("task", task)).strip() or task,
            steps=clean_steps,
            current_step=current_step,
            needs_visual_confirmation=needs_visual,
            cautions=clean_cautions,
        )

    # If the model returned unstructured plain text, map it safely
    lines = [line.strip("- *0123456789.").strip() for line in raw_text.splitlines() if line.strip()]
    return TaskGuidanceResponse(
        task=task,
        steps=lines if lines else [raw_text.strip()],
        current_step=lines[0] if lines else raw_text.strip(),
        needs_visual_confirmation=False,
        cautions=["Guidance was extracted from unstructured model response."],
    )


async def run_task_guidance(
    client: VisionModelClient,
    task: str,
    context: Optional[str] = None,
    previous_steps: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> TaskGuidanceResponse:
    clean_task = task.strip() if task else ""
    if not clean_task:
        raise HTTPException(
            status_code=422,
            detail="task cannot be empty or blank.",
        )

    prompt_sections = [
        TASK_GUIDANCE_SYSTEM_PROMPT,
        f"USER TASK: {clean_task}",
    ]
    if context and context.strip():
        prompt_sections.append(f"ADDITIONAL CONTEXT: {context.strip()}")
    if previous_steps and previous_steps.strip():
        prompt_sections.append(f"PREVIOUS STEPS COMPLETED / CURRENT PROGRESS: {previous_steps.strip()}")

    full_prompt = "\n\n".join(prompt_sections)

    try:
        if image_bytes and mime_type:
            raw_response = await client.generate_vision(
                prompt=full_prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                config=TASK_GUIDANCE_CONFIG,
            )
        else:
            raw_response = await client.generate_text(
                prompt=full_prompt,
                config=TASK_GUIDANCE_CONFIG,
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
            detail="The AI model could not process the guidance request. Please try again.",
        ) from exc

    return _parse_and_validate_guidance(raw_response, clean_task)


@router.post("/guide", response_model=TaskGuidanceResponse)
async def guide_task(
    task: str = Form(...),
    context: Optional[str] = Form(default=None),
    previous_steps: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    client: VisionModelClient = Depends(get_vision_client),
) -> TaskGuidanceResponse:
    clean_task = task.strip() if task else ""
    if not clean_task:
        raise HTTPException(
            status_code=422,
            detail="task cannot be empty or blank.",
        )

    # Process image if provided and non-empty
    has_image = image is not None and bool(image.filename and image.filename.strip())
    image_bytes = None
    mime_type = None
    if has_image:
        image_bytes, mime_type = await _validate_and_read_image(image)

    return await run_task_guidance(
        client=client,
        task=clean_task,
        context=context,
        previous_steps=previous_steps,
        image_bytes=image_bytes,
        mime_type=mime_type,
    )
