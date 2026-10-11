from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

from src.utils.config import (
    APP_NAME,
    APP_VERSION
)


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "AI-Powered Energy Consumption "
        "Optimization Assistant"
    )
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(router)


@app.get("/")
def root():

    return {
        "message":
            "AI-Powered Energy Consumption "
            "Optimization Assistant API",
        "version":
            APP_VERSION,
        "docs":
            "/docs"
    }