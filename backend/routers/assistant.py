import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from backend.routers.memory import get_memory_service
from backend.routers.tasks import run_task_guidance
from backend.routers.vision import get_vision_client
from backend.services.spatial_memory import SpatialMemoryService
from backend.services.speech import (
    MAX_TEXT_LENGTH,
    SpeechConfigurationError,
    SpeechEmptyResponseError,
    SpeechRateLimitError,
    SpeechService,
    SpeechServiceError,
    SpeechTimeoutError,
)
from backend.services.vision_model import (
    VisionModelClient,
    VisionModelConfigurationError,
    VisionModelEmptyResponseError,
    VisionModelError,
)

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])

VALID_INTENTS = {"auto", "scene", "read_text", "object", "task", "memory", "general"}

ASSISTANT_GENERAL_SYSTEM_PROMPT = """You are SAARTHI, an AI vision companion assisting a blind or low-vision user.
Respond to the user's message concisely, naturally, and warmly in 1 to 3 spoken sentences.

GUIDELINES:
1. Provide a direct, helpful, and natural response suitable for reading aloud.
2. Do not use Markdown formatting, headings, bullet lists, asterisks, or tables.
3. If the user asks you to describe visual surroundings or read text, explain that camera or image input is required.
4. Never claim that a walking route or physical area is safe or obstacle-free.
5. Do not provide medical dosages or hazardous instructions.
"""


class AssistantRequest(BaseModel):
    message: str
    intent: str = "auto"
    context: Optional[str] = None
    user_id: Optional[str] = None


class AssistantResponse(BaseModel):
    intent: str
    response: str
    action_required: bool
    required_input: Optional[str] = None
    cautions: list[str] = Field(default_factory=list)


GENERIC_OBJECT_WORDS = {
    "it",
    "this",
    "that",
    "something",
    "anything",
    "them",
    "me",
    "things",
    "stuff",
    "one",
}


def extract_target_object(message: str) -> Optional[str]:
    match = re.search(
        r"\b(?:find|look for|locate|search for|do you see|can you see|where is)\s+(?:my\s+|the\s+|a\s+|an\s+)?([a-zA-Z0-9\s]+?)(?:\?|\.|$|\bhere\b|\bin\b|\bon\b)",
        message,
        re.IGNORECASE,
    )
    if match:
        obj = match.group(1).strip()
        if obj and obj.lower() not in GENERIC_OBJECT_WORDS and len(obj.split()) <= 4:
            return obj
    return None


def classify_intent(message: str) -> str:
    msg = message.strip().lower()

    # Ambiguous or generic greetings / queries
    ambiguous_patterns = [
        r"^(?:hello|hi|hey|help|can you help me|can you help|check|what can you do|who are you|assistant)[\.?!]?$",
    ]
    for pattern in ambiguous_patterns:
        if re.match(pattern, msg):
            return "clarification"

    # 1. Read text intent
    read_text_patterns = [
        r"\b(read|transcribe|decipher)\b",
        r"\bwhat does (?:this|it|the text) say\b",
        r"\bwhat is written\b",
    ]
    for pattern in read_text_patterns:
        if re.search(pattern, msg):
            return "read_text"

    # 2. Scene intent
    scene_patterns = [
        r"\b(describe the scene|describe what|describe my surroundings|describe surroundings|describe the room)\b",
        r"\bwhat(?:'s| is) (?:in front of me|around me)\b",
        r"\b(what do you see|look around|tell me what you see)\b",
    ]
    for pattern in scene_patterns:
        if re.search(pattern, msg):
            return "scene"

    # 3. Memory intent
    memory_patterns = [
        r"\b(remember that|remember my|save memory|store memory|save to memory|add to memory|note that)\b",
        r"\b(where did i (?:put|leave|place))\b",
        r"\b(list my memories|show my memories|what memories|what did i save|what have i saved|my saved memories)\b",
    ]
    for pattern in memory_patterns:
        if re.search(pattern, msg):
            return "memory"

    # 4. Object assistance intent
    object_patterns = [
        r"\b(find my|look for my|locate my|search for my|do you see my|can you see my)\b",
        r"\b(find the|look for the|locate the|search for the|do you see the|can you see the)\b",
        r"\b(where is my|where is the)\b",
        r"\b(is there a|are there any)\b",
    ]
    for pattern in object_patterns:
        if re.search(pattern, msg):
            return "object"

    # 5. Task guidance intent
    task_patterns = [
        r"\b(help me complete this task|guide me|step by step|how do i|how to|how can i)\b",
        r"\b(help me (?:make|bake|brew|assemble|fix|prepare))\b",
        r"\b(steps to|instructions for|guide for)\b",
    ]
    for pattern in task_patterns:
        if re.search(pattern, msg):
            return "task"

    # 6. Fallback to general intent
    return "general"


def extract_user_id(request: AssistantRequest) -> Optional[str]:
    if request.user_id and request.user_id.strip():
        return request.user_id.strip()
    if request.context and request.context.strip():
        ctx = request.context.strip()
        match = re.search(r"\buser_id\s*[:=]\s*([a-zA-Z0-9_-]+)", ctx, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if len(ctx.split()) == 1 and not any(ch in ctx for ch in ":/?=\n"):
            return ctx
    return None


@router.post("/ask", response_model=AssistantResponse)
async def ask_assistant(
    request: AssistantRequest,
    client: VisionModelClient = Depends(get_vision_client),
    memory_service: SpatialMemoryService = Depends(get_memory_service),
) -> AssistantResponse:
    clean_message = request.message.strip() if request.message else ""
    if not clean_message:
        raise HTTPException(
            status_code=422,
            detail="message cannot be empty or blank.",
        )

    clean_intent = request.intent.strip().lower() if request.intent else "auto"
    if clean_intent not in VALID_INTENTS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid intent '{request.intent}'. Supported: {', '.join(sorted(VALID_INTENTS))}",
        )

    # Resolve capability
    if clean_intent == "auto":
        resolved_intent = classify_intent(clean_message)
    else:
        resolved_intent = clean_intent

    # Dispatch to capability
    if resolved_intent == "clarification":
        return AssistantResponse(
            intent="clarification",
            response="I can describe your surroundings, read text, locate everyday objects, guide you through tasks, or manage saved memories. What would you like help with?",
            action_required=True,
            required_input="clarification of desired task or capability",
            cautions=["Please specify a task or provide an image to proceed."],
        )

    if resolved_intent == "scene":
        return AssistantResponse(
            intent="scene",
            response="Please point your camera or share an image so I can describe the scene for you.",
            action_required=True,
            required_input="image",
            cautions=[
                "Scene descriptions require camera or image input.",
                "Never rely on scene descriptions alone for navigation safety.",
            ],
        )

    if resolved_intent == "read_text":
        return AssistantResponse(
            intent="read_text",
            response="Please point your camera at the text or provide an image so I can read it for you.",
            action_required=True,
            required_input="image",
            cautions=["Text reading requires an image showing the document, sign, or handwriting."],
        )

    if resolved_intent == "object":
        target = extract_target_object(clean_message)
        if target:
            return AssistantResponse(
                intent="object",
                response=f"Please point your camera or share an image so I can look for your {target}.",
                action_required=True,
                required_input="image",
                cautions=[
                    f"Looking for '{target}' requires an image of the area.",
                    "Visual detection does not guarantee obstacle-free access.",
                ],
            )
        return AssistantResponse(
            intent="object",
            response="Please tell me what object you are looking for and point your camera toward the area.",
            action_required=True,
            required_input="image and target_object",
            cautions=[
                "Object assistance requires both a target object name and an image.",
                "Visual detection does not guarantee obstacle-free access.",
            ],
        )

    if resolved_intent == "task":
        guidance = await run_task_guidance(
            client=client,
            task=clean_message,
            context=request.context,
        )
        task_label = guidance.task.rstrip(".")
        step_label = guidance.current_step.rstrip(".")
        response_text = f"For {task_label}: {step_label}."
        cautions = (
            guidance.cautions
            if guidance.cautions
            else ["Follow safe procedures and take your time."]
        )
        return AssistantResponse(
            intent="task",
            response=response_text,
            action_required=guidance.needs_visual_confirmation,
            required_input="image for visual confirmation" if guidance.needs_visual_confirmation else None,
            cautions=cautions,
        )

    if resolved_intent == "memory":
        user_id = extract_user_id(request)
        if not user_id:
            return AssistantResponse(
                intent="memory",
                response="Please provide your user identifier so I can access or save your spatial memories.",
                action_required=True,
                required_input="user_id",
                cautions=["Spatial memories are isolated per user identifier."],
            )

        msg_lower = clean_message.lower()

        # 1. Listing memories
        if any(w in msg_lower for w in ["list", "show my memories", "what memories", "what did i save", "what have i saved"]):
            memories = memory_service.list_memories(user_id=user_id)
            if memories:
                names = [m.name for m in memories]
                names_str = ", ".join(names[:5])
                if len(names) > 5:
                    names_str += f", and {len(names) - 5} more"
                return AssistantResponse(
                    intent="memory",
                    response=f"You have {len(memories)} saved memory items: {names_str}.",
                    action_required=False,
                    required_input=None,
                    cautions=["Spatial memory contains user-recorded landmarks."],
                )
            return AssistantResponse(
                intent="memory",
                response="You do not have any saved spatial memories yet.",
                action_required=False,
                required_input=None,
                cautions=["Spatial memory is currently empty."],
            )

        # 2. Saving a memory
        save_match = re.search(
            r"^(?:please\s+)?(?:remember\s+(?:that\s+)?(?:my\s+)?|save\s+(?:memory\s+)?|store\s+(?:memory\s+)?|note\s+(?:that\s+)?)(.+)$",
            clean_message,
            re.IGNORECASE,
        )
        if save_match:
            detail = save_match.group(1).strip()
            # Split item name and location/description if predicate is found
            pred_match = re.search(r"^(.+?)\s+(?:is|are)\s+(.+)$", detail, re.IGNORECASE)
            if pred_match:
                item_name = pred_match.group(1).strip()
                item_desc = pred_match.group(2).strip()
            else:
                prep_match = re.search(r"^(.+?)\s+(in|on|at|by|under|beside)\s+(.+)$", detail, re.IGNORECASE)
                if prep_match:
                    item_name = prep_match.group(1).strip()
                    item_desc = f"{prep_match.group(2).strip()} {prep_match.group(3).strip()}"
                else:
                    item_name = detail[:30].strip()
                    item_desc = detail

            memory_service.create_memory(
                user_id=user_id,
                name=item_name,
                description=item_desc,
            )
            return AssistantResponse(
                intent="memory",
                response=f"I have saved that your {item_name} is {item_desc}.",
                action_required=False,
                required_input=None,
                cautions=["Spatial memory stores textual notes, not live location tracking."],
            )

        # 3. Recalling a memory
        recall_match = re.search(
            r"(?:where did i\s+(?:put|leave|place)\s+(?:my\s+|the\s+)?|where is\s+(?:my\s+|the\s+)?)(.+?)[\.?!]?$",
            clean_message,
            re.IGNORECASE,
        )
        item_to_find = recall_match.group(1).strip() if recall_match else clean_message.strip()

        memories = memory_service.list_memories(user_id=user_id)
        matched_mem = None
        for mem in memories:
            if item_to_find.lower() in mem.name.lower() or mem.name.lower() in item_to_find.lower():
                matched_mem = mem
                break

        if matched_mem:
            return AssistantResponse(
                intent="memory",
                response=f"According to your saved memory, your {matched_mem.name} is {matched_mem.description}.",
                action_required=False,
                required_input=None,
                cautions=["Saved memory reflects your previously recorded note, not live tracking."],
            )

        return AssistantResponse(
            intent="memory",
            response=f"I checked your saved memories, but could not find a record for {item_to_find}.",
            action_required=False,
            required_input=None,
            cautions=["No matching memory record was found."],
        )

    # General intent
    prompt_parts = [
        ASSISTANT_GENERAL_SYSTEM_PROMPT,
        f"USER MESSAGE: {clean_message}",
    ]
    if request.context and request.context.strip():
        prompt_parts.append(f"CONTEXT: {request.context.strip()}")

    full_prompt = "\n\n".join(prompt_parts)

    try:
        raw_response = await client.generate_text(prompt=full_prompt)
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
            detail="The AI model could not process the assistant request. Please try again.",
        ) from exc

    clean_resp = raw_response.strip().replace("*", "").replace("#", "")
    return AssistantResponse(
        intent="general",
        response=clean_resp,
        action_required=False,
        required_input=None,
        cautions=["General AI responses should not be relied upon for safety-critical navigation or medical decisions."],
    )


class SpeakRequest(BaseModel):
    text: str


def get_speech_service() -> SpeechService:
    return SpeechService()


@router.post("/speak")
async def speak_text(
    request: SpeakRequest,
    service: SpeechService = Depends(get_speech_service),
) -> Response:
    clean_text = request.text.strip() if request.text else ""
    if not clean_text:
        raise HTTPException(
            status_code=422,
            detail="text cannot be empty or blank.",
        )
    if len(clean_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters.",
        )

    try:
        audio_bytes, mime_type = await service.generate_speech(text=clean_text)
    except SpeechConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI credentials are not configured. Contact the server administrator.",
        ) from exc
    except SpeechRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The speech generation rate limit was exceeded. Please try again shortly.",
        ) from exc
    except SpeechTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The speech generation request timed out. Please try again.",
        ) from exc
    except SpeechEmptyResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI speech model returned an empty response. Please try again.",
        ) from exc
    except SpeechServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI speech model could not process the request. Please try again.",
        ) from exc

    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
        },
    )
