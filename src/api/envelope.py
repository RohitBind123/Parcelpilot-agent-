"""One response shape, and one error shape.

The Day-1 checklist item deferred to M8. Every route returns the same envelope,
so a client has one thing to unwrap and one place to look for a failure, rather
than a per-route guess at whether the body is the payload or a wrapper round it.

Errors carry a stable `code` alongside the message. The message is for a person
and may be reworded; the code is what a client branches on, and rewording an
error should not break a client that was handling it.
"""

from __future__ import annotations

from typing import Any, Final

from fastapi import HTTPException
from fastapi.responses import JSONResponse


class ErrorCode:
    UNAUTHENTICATED: Final = "unauthenticated"
    FORBIDDEN: Final = "forbidden"
    NOT_FOUND: Final = "not_found"
    INVALID_REQUEST: Final = "invalid_request"
    CONFLICT: Final = "conflict"
    INTERNAL: Final = "internal"


def ok(data: Any = None, **meta: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None, **({"meta": meta} if meta else {})}


def failed(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


class ApiError(HTTPException):
    """An HTTPException that renders into the envelope.

    Raised rather than returned so a helper deep in a route can refuse without
    every caller between it and the response having to thread the failure back
    by hand.
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code

    @classmethod
    def unauthenticated(cls, message: str = "sign in to continue") -> ApiError:
        return cls(401, ErrorCode.UNAUTHENTICATED, message)

    @classmethod
    def not_found(cls, message: str) -> ApiError:
        return cls(404, ErrorCode.NOT_FOUND, message)

    @classmethod
    def invalid(cls, message: str) -> ApiError:
        return cls(400, ErrorCode.INVALID_REQUEST, message)

    @classmethod
    def conflict(cls, message: str) -> ApiError:
        return cls(409, ErrorCode.CONFLICT, message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content=failed(self.code, str(self.detail))
        )


__all__ = ["ApiError", "ErrorCode", "failed", "ok"]
