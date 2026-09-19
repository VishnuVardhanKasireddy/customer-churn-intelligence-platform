from fastapi import FastAPI
from src.api.routes.prediction import router as prediction_router

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="API for predicting customer churn using behavioral features.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "customer-churn-intelligence-api",
    }

app.include_router(prediction_router)