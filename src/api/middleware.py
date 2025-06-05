# src/api/middleware.py
from fastapi import Request, Response

async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update({
        "Content-Security-Policy": "default-src 'self' resume-screener-v3kw.onrender.com",
        "X-Content-Type-Options": "nosniff"
    })
    return response