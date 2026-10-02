"""Bearer token authentication middleware for FastMCP."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_UNPROTECTED = {"/health", "/", "/docs", "/openapi.json"}


class BearerAuth(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        if request.url.path in _UNPROTECTED:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != self.api_key:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)
