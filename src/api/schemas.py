from pydantic import BaseModel,Field

class ChurnPredictionRequest(BaseModel):
    total_orders: int = Field(..., ge=0)
    total_spend: float = Field(..., ge=0)
    total_quantity: int = Field(..., ge=0)
    unique_products: int = Field(..., ge=0)
    active_days: int = Field(..., ge=0)
    average_order_value: float = Field(..., ge=0)

    customer_lifespan_days: int = Field(..., ge=0)
    recency_days: int = Field(..., ge=0)
    average_interpurchase_days: float = Field(..., ge=0)
    median_interpurchase_days: float = Field(..., ge=0)

    orders_30d: int = Field(..., ge=0)
    orders_60d: int = Field(..., ge=0)
    orders_90d: int = Field(..., ge=0)

    spend_30d: float = Field(..., ge=0)
    spend_60d: float = Field(..., ge=0)
    spend_90d: float = Field(..., ge=0)

    orders_previous_90d: int = Field(..., ge=0)
    spend_previous_90d: float = Field(..., ge=0)

    order_trend_ratio: float = Field(..., ge=0)
    spend_trend_ratio: float = Field(..., ge=0)


class ChurnPredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., ge=0, le=1)
    risk_level: str