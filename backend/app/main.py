from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    print("=" * 60)
    print(f"{settings.PROJECT_NAME} started successfully")
    print("=" * 60)

    yield

    print("=" * 60)
    print("Server shutting down...")
    print("=" * 60)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "Running",
        "edge_ai": True
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "server": "online"
    }