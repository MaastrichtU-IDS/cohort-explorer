"""Admin API over the LLM mapping decision store (CohortVarLinker/src/llm_decisions.py).

  GET /api/llm-decisions/stats
  GET /api/llm-decisions/runs
  GET /api/llm-decisions/decisions?run_id=&status=&source=&target=&model=&q=&cached=&limit=&offset=
  GET /api/llm-decisions/decision/{id}          full record: prompts and raw response included
  GET /api/llm-decisions/pair/{pair_key}        every decision for one pair
  GET /api/llm-decisions/consistency?only_conflicts=true
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.auth import get_current_user
from src.config import settings

router = APIRouter()


def _require_admin(user: dict) -> None:
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")


def _store():
    from CohortVarLinker.src import llm_decisions
    return llm_decisions


@router.get("/llm-decisions/stats")
async def llm_decision_stats(user: Any = Depends(get_current_user)):
    _require_admin(user)
    return _store().stats()


@router.get("/llm-decisions/runs")
async def llm_decision_runs(limit: int = Query(200, le=2000), user: Any = Depends(get_current_user)):
    _require_admin(user)
    return {"items": _store().list_runs(limit)}


@router.get("/llm-decisions/decisions")
async def llm_decisions_list(
    run_id: Optional[str] = None, status: Optional[str] = None, source: Optional[str] = None,
    target: Optional[str] = None, model: Optional[str] = None, q: Optional[str] = None,
    cached: Optional[bool] = None, limit: int = Query(100, le=1000), offset: int = Query(0, ge=0),
    user: Any = Depends(get_current_user),
):
    _require_admin(user)
    return _store().list_decisions(run_id=run_id, status=status, source=source, target=target, model=model,
                                   q=q, cached=cached, limit=limit, offset=offset)


@router.get("/llm-decisions/decision/{decision_id}")
async def llm_decision_detail(decision_id: int, user: Any = Depends(get_current_user)):
    _require_admin(user)
    row = _store().get_decision(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail="No such decision.")
    return row


@router.get("/llm-decisions/pair/{pair_key}")
async def llm_decision_pair(pair_key: str, user: Any = Depends(get_current_user)):
    _require_admin(user)
    return {"pair_key": pair_key, "items": _store().pair_history(pair_key)}


@router.get("/llm-decisions/consistency")
async def llm_decision_consistency(only_conflicts: bool = True, min_decisions: int = Query(2, ge=2),
                                   limit: int = Query(500, le=5000), user: Any = Depends(get_current_user)):
    _require_admin(user)
    return _store().consistency(min_decisions=min_decisions, only_conflicts=only_conflicts, limit=limit)
