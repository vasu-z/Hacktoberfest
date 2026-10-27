from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.services.spatial_memory import (
    MemoryCreateRequest,
    MemoryRecord,
    MemoryUpdateRequest,
    SpatialMemoryService,
)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


def get_memory_service() -> SpatialMemoryService:
    return SpatialMemoryService()


@router.post("", response_model=MemoryRecord, status_code=status.HTTP_201_CREATED)
async def create_memory(
    request: MemoryCreateRequest,
    service: SpatialMemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    clean_user = request.user_id.strip() if request.user_id else ""
    clean_name = request.name.strip() if request.name else ""
    clean_desc = request.description.strip() if request.description else ""

    if not clean_user:
        raise HTTPException(
            status_code=422,
            detail="user_id cannot be empty or blank.",
        )
    if not clean_name:
        raise HTTPException(
            status_code=422,
            detail="name cannot be empty or blank.",
        )
    if not clean_desc:
        raise HTTPException(
            status_code=422,
            detail="description cannot be empty or blank.",
        )

    clean_location = request.location_label.strip() if request.location_label else None

    return service.create_memory(
        user_id=clean_user,
        name=clean_name,
        description=clean_desc,
        location_label=clean_location,
    )


@router.get("", response_model=list[MemoryRecord])
async def list_memories(
    user_id: str = Query(..., description="The user identifier to list memories for"),
    service: SpatialMemoryService = Depends(get_memory_service),
) -> list[MemoryRecord]:
    clean_user = user_id.strip() if user_id else ""
    if not clean_user:
        raise HTTPException(
            status_code=422,
            detail="user_id query parameter cannot be empty or blank.",
        )
    return service.list_memories(user_id=clean_user)


@router.get("/{memory_id}", response_model=MemoryRecord)
async def get_memory(
    memory_id: str,
    user_id: str = Query(..., description="The user identifier"),
    service: SpatialMemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    clean_user = user_id.strip() if user_id else ""
    if not clean_user:
        raise HTTPException(
            status_code=422,
            detail="user_id query parameter cannot be empty or blank.",
        )

    memory = service.get_memory(memory_id=memory_id, user_id=clean_user)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory record not found.",
        )
    return memory


@router.patch("/{memory_id}", response_model=MemoryRecord)
async def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    service: SpatialMemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    clean_user = request.user_id.strip() if request.user_id else ""
    if not clean_user:
        raise HTTPException(
            status_code=422,
            detail="user_id cannot be empty or blank.",
        )

    fields_to_update = request.model_fields_set - {"user_id"}
    if not fields_to_update:
        raise HTTPException(
            status_code=422,
            detail="At least one field to update (name, description, location_label) must be provided.",
        )

    new_name = None
    if "name" in fields_to_update:
        if request.name is None or not request.name.strip():
            raise HTTPException(
                status_code=422,
                detail="name cannot be empty or blank when updated.",
            )
        new_name = request.name.strip()

    new_desc = None
    if "description" in fields_to_update:
        if request.description is None or not request.description.strip():
            raise HTTPException(
                status_code=422,
                detail="description cannot be empty or blank when updated.",
            )
        new_desc = request.description.strip()

    new_location = None
    clear_location = False
    if "location_label" in fields_to_update:
        if request.location_label is None or not request.location_label.strip():
            clear_location = True
        else:
            new_location = request.location_label.strip()

    updated = service.update_memory(
        memory_id=memory_id,
        user_id=clean_user,
        name=new_name,
        description=new_desc,
        location_label=new_location,
        clear_location_label=clear_location,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory record not found.",
        )
    return updated


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: str = Query(..., description="The user identifier"),
    service: SpatialMemoryService = Depends(get_memory_service),
) -> dict[str, str]:
    clean_user = user_id.strip() if user_id else ""
    if not clean_user:
        raise HTTPException(
            status_code=422,
            detail="user_id query parameter cannot be empty or blank.",
        )

    deleted = service.delete_memory(memory_id=memory_id, user_id=clean_user)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory record not found.",
        )
    return {"status": "deleted", "id": memory_id}
