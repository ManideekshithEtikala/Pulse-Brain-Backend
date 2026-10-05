"""Standalone FastAPI entrypoint for the Pulse-Brain agents."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers.brain_agent_routes import router as brain_agent_router


app = FastAPI(title="Pulse-Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(brain_agent_router, prefix="/company-brain")


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok", "service": "pulse-brain"}
