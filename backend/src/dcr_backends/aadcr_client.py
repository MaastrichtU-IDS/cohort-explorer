"""Bounded asynchronous HTTP client for the native AADCR v2 API."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from src.config import Settings
from src.dcr_backends.aadcr_auth import mint_aadcr_token

_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")
_AUTH_PATTERN = re.compile(r"(?i)authorization\s*[:=]\s*[^;,\s]+(?:\s+[^;,\s]+)?")
_COOKIE_PATTERN = re.compile(r"(?i)(?:set-)?cookie\s*[:=]\s*[^;,\s]+")


class AadcrUpstreamError(RuntimeError):
    """Safe error boundary for an unsuccessful AADCR request."""

    def __init__(
        self,
        *,
        method: str,
        path: str,
        detail: str,
        failed_step: str,
        status_code: int | None,
        retryable: bool,
    ):
        self.method = method
        self.path = path
        self.detail = detail
        self.failed_step = failed_step
        self.status_code = status_code
        self.retryable = retryable
        status = str(status_code) if status_code is not None else "connection-error"
        super().__init__(
            f"AADCR request failed during {failed_step}: {method} {path} "
            f"returned {status}: {detail}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "provider": "aadcrv2",
            "failed_step": self.failed_step,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "retryable": self.retryable,
        }

    def __repr__(self) -> str:
        return (
            "AadcrUpstreamError("
            f"method={self.method!r}, path={self.path!r}, status_code={self.status_code!r}, "
            f"failed_step={self.failed_step!r}, detail={self.detail!r}, retryable={self.retryable!r})"
        )


class AadcrClient:
    """One short-lived authenticated client per logical adapter operation."""

    def __init__(
        self,
        settings: Settings,
        user: dict[str, Any],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._settings = settings
        self._user_email = str(user.get("email") or "").strip().lower()
        token = mint_aadcr_token(user, settings)
        self._authorization_value = f"Bearer {token}"
        timeout = httpx.Timeout(settings.aadcrv2_timeout_seconds)
        self._client = httpx.AsyncClient(
            base_url=settings.aadcrv2_url,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    async def __aenter__(self) -> AadcrClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"AadcrClient(base_url={self._settings.aadcrv2_url!r}, "
            f"user={self._user_email!r}, timeout={self._settings.aadcrv2_timeout_seconds!r})"
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        self._authorization_value = ""

    @staticmethod
    def _validate_path(path: str) -> tuple[str, str]:
        parsed = urlsplit(path)
        decoded_path = parsed.path
        for _decode_pass in range(3):
            next_path = unquote(decoded_path)
            if next_path == decoded_path:
                break
            decoded_path = next_path
        path_segments = decoded_path.split("/")
        if (
            parsed.scheme
            or parsed.netloc
            or not decoded_path.startswith("/api/")
            or "\\" in decoded_path
            or any(segment in {".", ".."} for segment in path_segments)
        ):
            raise ValueError("AADCR requests require a safe relative API path under /api/")
        return path, parsed.path

    def _sanitize_detail(self, detail: object) -> str:
        if not isinstance(detail, str) or not detail.strip():
            return "AADCR v2 returned an unsuccessful response"
        sanitized = detail.strip()
        for secret in (self._settings.aadcrv2_jwt_secret, self._authorization_value):
            if secret:
                sanitized = sanitized.replace(secret, "[REDACTED]")
        sanitized = _BEARER_PATTERN.sub("Bearer [REDACTED]", sanitized)
        sanitized = _AUTH_PATTERN.sub("Authorization: [REDACTED]", sanitized)
        sanitized = _COOKIE_PATTERN.sub("Cookie: [REDACTED]", sanitized)
        return sanitized[:300]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        failed_step: str,
        **request_kwargs: Any,
    ) -> httpx.Response:
        request_path, safe_path = self._validate_path(path)
        normalized_method = method.upper()
        headers = dict(request_kwargs.pop("headers", {}) or {})
        headers["Authorization"] = self._authorization_value
        headers.setdefault("Accept", "application/json")
        connection_failed = False
        try:
            response = await self._client.request(
                normalized_method,
                request_path,
                headers=headers,
                **request_kwargs,
            )
        except httpx.RequestError:
            connection_failed = True
        if connection_failed:
            raise AadcrUpstreamError(
                method=normalized_method,
                path=safe_path,
                detail="AADCR v2 backend is unreachable",
                failed_step=failed_step,
                status_code=None,
                retryable=True,
            )

        if response.status_code >= 400:
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = None
            detail = response_payload.get("detail") if isinstance(response_payload, dict) else None
            raise AadcrUpstreamError(
                method=normalized_method,
                path=safe_path,
                detail=self._sanitize_detail(detail),
                failed_step=failed_step,
                status_code=response.status_code,
                retryable=response.status_code >= 500,
            )
        return response

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        failed_step: str,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        response = await self._request(
            method,
            path,
            failed_step=failed_step,
            json=json_body,
            params=params,
            data=data,
            files=files,
        )
        if response.status_code == 204 or not response.content:
            return None
        invalid_json = False
        try:
            payload = response.json()
        except ValueError:
            invalid_json = True
        if invalid_json:
            raise AadcrUpstreamError(
                method=method.upper(),
                path=urlsplit(path).path,
                detail="AADCR v2 did not return valid JSON",
                failed_step=failed_step,
                status_code=response.status_code,
                retryable=False,
            )
        return payload

    async def request_bytes(
        self,
        method: str,
        path: str,
        *,
        failed_step: str,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        response = await self._request(
            method,
            path,
            failed_step=failed_step,
            params=params,
        )
        return response.content

    async def upload_file(
        self,
        path: str,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        failed_step: str,
        form: dict[str, Any] | None = None,
        field_name: str = "file",
    ) -> Any:
        return await self.request_json(
            "POST",
            path,
            failed_step=failed_step,
            data=form,
            files={field_name: (filename, content, content_type)},
        )
