from fastapi import APIRouter
from src.api.schemas import (ChurnPredictionRequest,ChurnPredictionResponse)
from src.model.predict import predict_churn


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)

@router.post(
    "/predict",
    response_model=ChurnPredictionResponse
)
def predict(request:ChurnPredictionRequest):
    result = predict_churn(request.model_dump())
    return result


