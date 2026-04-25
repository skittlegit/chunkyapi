"""FastAPI application entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import cases as cases_routes
from .routes import health as health_routes
from .routes import planning as planning_routes
from .routes import simulation as simulation_routes
from .routes import validation as validation_routes


logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(title="ChunkyAPI", version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_routes.router)
app.include_router(planning_routes.router)
app.include_router(simulation_routes.router)
app.include_router(validation_routes.router)
app.include_router(cases_routes.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "ChunkyAPI",
        "version": settings.version,
        "docs": "/docs",
    }
