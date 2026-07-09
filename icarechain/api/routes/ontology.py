from typing import Optional

from fastapi import APIRouter, Query
from pydantic import EmailStr

from api.models.duo import (
    MODIFIER_LABELS,
    MODIFIER_VALUES,
    PERMISSION_HIERARCHY,
    PERMISSION_LABELS,
    PERMISSION_VALUES,
)
from api.services.ontology import get_ontology_service

router = APIRouter(prefix="/ontology", tags=["ontology"])

wizard_router = router

@router.get("/permissions", summary="Full DUO permission and modifier ontology (codes, labels, parents, bitmasks)")
async def permissions():
    perms = []
    for code in PERMISSION_VALUES:
        parent = next((p for p, kids in PERMISSION_HIERARCHY.items() if code in kids), None)
        perms.append({
            "code": code,
            "label": PERMISSION_LABELS.get(code, code),
            "parent": parent,
            "children": PERMISSION_HIERARCHY.get(code, []),
        })
    mods = [
        {"code": code, "label": MODIFIER_LABELS.get(code, code), "bitmask": MODIFIER_VALUES[code]}
        for code in MODIFIER_VALUES
    ]
    return {"permissions": perms, "modifiers": mods}

@router.get("/diseases", summary="Search the supported ICD-10 disease codes by free text or code")
async def search_diseases(q: str = Query(..., min_length=2), limit: int = Query(10, ge=1, le=50)):
    results = await get_ontology_service().search_diseases(q, limit=limit)
    return {"query": q, "results": results}

@router.get("/diseases/{code:path}", summary="Validate / look up a single ICD-10 code (any level)")
async def get_disease(code: str):
    return await get_ontology_service().validate_disease(code)

@router.get("/institutions", summary="Search ROR institution registry by free text (optional country filter)")
async def search_institutions(
    q: str = Query(..., min_length=2),
    country: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
):
    results = await get_ontology_service().search_institutions(q, country=country, limit=limit)
    return {"query": q, "results": results}

@router.get("/institutions/{ror_id}", summary="Validate / look up a single ROR institution id")
async def get_institution(ror_id: str):
    return await get_ontology_service().validate_institution(ror_id)
