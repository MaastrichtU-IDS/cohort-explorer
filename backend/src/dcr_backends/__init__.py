"""Provider-neutral Data Clean Room backend interfaces."""

from src.dcr_backends.contracts import DcrBackend
from src.dcr_backends.factory import get_dcr_backend
from src.dcr_backends.models import (
    DcrCapabilities,
    DcrListResult,
    DcrRoom,
    LiveCreateResult,
    ProviderError,
)

__all__ = [
    "DcrBackend",
    "DcrCapabilities",
    "DcrListResult",
    "DcrRoom",
    "LiveCreateResult",
    "ProviderError",
    "get_dcr_backend",
]
