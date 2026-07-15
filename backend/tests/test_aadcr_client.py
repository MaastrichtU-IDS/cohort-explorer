import asyncio
import json

import httpx
import pytest

from src.dcr_backends.aadcr_client import AadcrClient, AadcrUpstreamError

TEST_AADCR_SECRET = "shared-test-secret"


def run(coroutine):
    return asyncio.run(coroutine)


def client_settings(settings_factory):
    return settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_url="http://aadcr.test",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_timeout_seconds=7.5,
    )


def test_client_reuses_one_bearer_token_per_logical_operation(settings_factory, monkeypatch):
    minted_for: list[str] = []

    def fake_mint(user, _settings):
        minted_for.append(user["email"])
        return "short-lived-operation-token"

    monkeypatch.setattr("src.dcr_backends.aadcr_client.mint_aadcr_token", fake_mint)
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request):
        observed.append(request)
        return httpx.Response(200, json={"ok": True})

    async def exercise():
        async with AadcrClient(
            client_settings(settings_factory),
            {"email": "Owner@Example.Test"},
            transport=httpx.MockTransport(handler),
        ) as client:
            assert await client.request_json("GET", "/api/dcr/", failed_step="list rooms") == {"ok": True}
            assert await client.request_json("GET", "/api/system/info", failed_step="system info") == {"ok": True}

    run(exercise())

    assert minted_for == ["Owner@Example.Test"]
    assert [request.headers["Authorization"] for request in observed] == [
        "Bearer short-lived-operation-token",
        "Bearer short-lived-operation-token",
    ]
    assert observed[0].extensions["timeout"]["connect"] == 7.5
    assert observed[0].extensions["timeout"]["read"] == 7.5


def test_client_handles_json_and_multipart_without_retaining_payload(settings_factory):
    requests: list[tuple[str, str, bytes, str]] = []

    def handler(request: httpx.Request):
        requests.append(
            (
                request.method,
                request.url.path,
                request.content,
                request.headers.get("content-type", ""),
            )
        )
        return httpx.Response(200, json={"id": "created"})

    async def exercise():
        async with AadcrClient(
            client_settings(settings_factory),
            {"email": "owner@example.test"},
            transport=httpx.MockTransport(handler),
        ) as client:
            created = await client.request_json(
                "POST",
                "/api/dcr/",
                json_body={"name": "Synthetic cohort room"},
                failed_step="create room",
            )
            uploaded = await client.upload_file(
                "/api/upload",
                filename="synthetic.csv",
                content=b"subject_id,age\nSYN-001,72\n",
                content_type="text/csv",
                form={"description": "metadata-only synthetic fixture"},
                failed_step="upload synthetic dataset",
            )
            assert created == uploaded == {"id": "created"}
            assert "subject_id" not in repr(client)

    run(exercise())

    assert json.loads(requests[0][2]) == {"name": "Synthetic cohort room"}
    assert requests[1][0:2] == ("POST", "/api/upload")
    assert "multipart/form-data" in requests[1][3]
    assert b"synthetic.csv" in requests[1][2]
    assert b"SYN-001" in requests[1][2]


@pytest.mark.parametrize("status_code", [400, 403, 500, 503])
def test_upstream_http_errors_are_normalized_and_redacted(settings_factory, status_code):
    secret = TEST_AADCR_SECRET

    def handler(_request: httpx.Request):
        return httpx.Response(
            status_code,
            json={"detail": f"failure {secret}; Authorization: Bearer upstream-token; Cookie: private"},
        )

    async def exercise():
        async with AadcrClient(
            client_settings(settings_factory),
            {"email": "owner@example.test", "access_token": "browser-token"},
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.request_json(
                "POST",
                "/api/dcr/room-1/dev/data-nodes?private=query",
                json_body={"schema": "sensitive-row-schema"},
                failed_step="create data node",
            )

    with pytest.raises(AadcrUpstreamError) as caught:
        run(exercise())

    error = caught.value
    assert error.method == "POST"
    assert error.path == "/api/dcr/room-1/dev/data-nodes"
    assert error.status_code == status_code
    assert error.failed_step == "create data node"
    assert error.retryable is (status_code >= 500)
    rendered = f"{error!s} {error!r} {error.to_dict()}"
    for sensitive in (secret, "upstream-token", "private", "browser-token", "sensitive-row-schema"):
        assert sensitive not in rendered


def test_malformed_json_and_connection_failure_are_normalized(settings_factory):
    settings = client_settings(settings_factory)

    async def malformed():
        async with AadcrClient(
            settings,
            {"email": "owner@example.test"},
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=b"not-json-browser-token",
                    headers={"content-type": "text/plain"},
                )
            ),
        ) as client:
            await client.request_json("GET", "/api/dcr/room-1", failed_step="read room")

    def connection_failure(request: httpx.Request):
        raise httpx.ConnectError("dial failed with browser-token", request=request)

    async def disconnected():
        async with AadcrClient(
            settings,
            {"email": "owner@example.test", "access_token": "browser-token"},
            transport=httpx.MockTransport(connection_failure),
        ) as client:
            await client.upload_file(
                "/api/upload",
                filename="sensitive.csv",
                content=b"subject_id\nSENSITIVE-SUBJECT\n",
                content_type="text/csv",
                failed_step="upload dataset",
            )

    with pytest.raises(AadcrUpstreamError, match="valid JSON") as malformed_error:
        run(malformed())
    assert malformed_error.value.status_code == 200
    assert malformed_error.value.retryable is False
    assert malformed_error.value.__cause__ is None
    assert malformed_error.value.__context__ is None

    with pytest.raises(AadcrUpstreamError, match="unreachable") as connection_error:
        run(disconnected())
    assert connection_error.value.status_code is None
    assert connection_error.value.retryable is True
    assert connection_error.value.__cause__ is None
    assert connection_error.value.__context__ is None
    rendered = f"{connection_error.value!s} {connection_error.value!r} {vars(connection_error.value)}"
    assert "browser-token" not in rendered
    assert "SENSITIVE-SUBJECT" not in rendered


@pytest.mark.parametrize(
    "path",
    [
        "https://attacker.test/api/dcr",
        "//attacker.test/api/dcr",
        "/api/../secret",
        "/api/%2e%2e/secret",
        "/api/dcr\\..\\secret",
    ],
)
def test_client_rejects_unsafe_paths_before_sending(settings_factory, path):
    sent = False

    def handler(_request: httpx.Request):
        nonlocal sent
        sent = True
        return httpx.Response(200, json={})

    async def exercise():
        async with AadcrClient(
            client_settings(settings_factory),
            {"email": "owner@example.test"},
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.request_json("GET", path, failed_step="unsafe request")

    with pytest.raises(ValueError, match="relative API path"):
        run(exercise())
    assert sent is False
