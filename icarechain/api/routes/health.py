from fastapi import APIRouter

from api.config import get_settings
from api.services.blockchain import get_blockchain_service
from api.services.cache import get_cache

router = APIRouter(tags=["health"])

@router.get("/health", summary="Liveness probe: API, blockchain RPC, and cache status")
async def health():
    settings = get_settings()
    cache = get_cache()

    connected = False
    chain_id = None
    try:
        service = get_blockchain_service()
        connected = service.w3.is_connected()
        if connected:
            chain_id = service.w3.eth.chain_id
    except Exception:
        pass

    return {
        "status": "ok",
        "version": "3.0.0",
        "blockchain": {"connected": connected, "chain_id": chain_id, "rpc_url": settings.rpc_url},
        "cache": cache.get_stats(),
    }
