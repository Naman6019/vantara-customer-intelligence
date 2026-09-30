from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import get_logger
from api.routers.predictions import router as predictions_router

logger = get_logger("api_main")

app = FastAPI(
    title="Vantara Retail Solutions — Customer Behavior Prediction API",
    description="Machine Learning & Deep Learning REST API for Churn Risk, Customer Lifetime Value (CLV), Segmentation, and Explainability (XAI).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit / web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions_router, tags=["Customer Predictions"])


@app.get("/")
def root():
    return {
        "message": "Welcome to Vantara Customer Behavior Prediction Platform API",
        "documentation": "/docs",
        "health_check": "/health",
        "metadata": "/metadata",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
