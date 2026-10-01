"""FastAPI application exposing the trip-duration model.

Phase 0 ships only ``GET /health``, which is enough to prove the production
image builds and runs. ``POST /predict`` arrives in Phase 6.
"""

from typing import Final

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

#: Set once a trained model is loaded at startup (Phase 6).
MODEL_VERSION: Final[str | None] = None

app = FastAPI(title="NYC trip ETA", version="0.1.0")


class Health(BaseModel):
    """Service liveness payload."""

    # pydantic reserves the ``model_`` prefix for its own API, so a field called
    # ``model_version`` needs the protected namespace explicitly cleared.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str | None


@app.get("/health")
def health() -> Health:
    """Report service liveness and which model version is loaded."""
    return Health(status="ok", model_version=MODEL_VERSION)
