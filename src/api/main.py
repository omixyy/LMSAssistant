from __future__ import annotations

from fastapi import FastAPI

from api.routers import grade_router, health_router


def create_app() -> FastAPI:
    app = FastAPI(title="LMSAssistant API")
    app.include_router(health_router)
    app.include_router(grade_router)
    return app


app = create_app()

