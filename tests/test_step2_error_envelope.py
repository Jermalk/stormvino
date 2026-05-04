"""
Step 2 tests — Anthropic error envelope handler.
Uses FastAPI TestClient; no GPU required.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from anthropic_layer import AnthropicRequest


def make_app_with_handler():
    """Build a minimal FastAPI app with just the error envelope handler."""
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        if request.url.path.startswith("/v1/messages"):
            error_type = {
                401: "authentication_error",
                400: "invalid_request_error",
                404: "not_found_error",
                429: "rate_limit_error",
            }.get(exc.status_code, "api_error")
            return JSONResponse(status_code=exc.status_code, content={
                "type":  "error",
                "error": {"type": error_type, "message": str(exc.detail)},
            })
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.post("/v1/messages")
    async def fake_messages():
        raise HTTPException(status_code=400, detail="bad request")

    @app.post("/v1/chat/completions")
    async def fake_chat():
        raise HTTPException(status_code=400, detail="bad request")

    return app


client = TestClient(make_app_with_handler(), raise_server_exceptions=False)


class TestErrorEnvelope:
    def test_messages_path_returns_anthropic_format(self):
        r = client.post("/v1/messages", json={})
        assert r.status_code == 400
        body = r.json()
        assert body["type"] == "error"
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert body["error"]["type"] == "invalid_request_error"

    def test_messages_401_maps_to_authentication_error(self):
        app = FastAPI()

        @app.exception_handler(HTTPException)
        async def handler(request: Request, exc: HTTPException):
            if request.url.path.startswith("/v1/messages"):
                error_type = {401: "authentication_error", 400: "invalid_request_error"}.get(exc.status_code, "api_error")
                return JSONResponse(status_code=exc.status_code, content={"type": "error", "error": {"type": error_type, "message": str(exc.detail)}})
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        @app.post("/v1/messages")
        async def ep():
            raise HTTPException(status_code=401, detail="unauthorized")

        c = TestClient(app, raise_server_exceptions=False)
        r = c.post("/v1/messages", json={})
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"

    def test_non_messages_path_returns_fastapi_format(self):
        r = client.post("/v1/chat/completions", json={})
        assert r.status_code == 400
        body = r.json()
        assert "detail" in body
        assert "type" not in body

    def test_messages_count_tokens_path_also_gets_anthropic_format(self):
        app = FastAPI()

        @app.exception_handler(HTTPException)
        async def handler(request: Request, exc: HTTPException):
            if request.url.path.startswith("/v1/messages"):
                return JSONResponse(status_code=exc.status_code, content={"type": "error", "error": {"type": "api_error", "message": str(exc.detail)}})
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        @app.post("/v1/messages/count_tokens")
        async def ep():
            raise HTTPException(status_code=400, detail="bad")

        c = TestClient(app, raise_server_exceptions=False)
        r = c.post("/v1/messages/count_tokens", json={})
        assert r.json()["type"] == "error"
