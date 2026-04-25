from fastapi import APIRouter

from ..config import settings
from ..models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.version)
