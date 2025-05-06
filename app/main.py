from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="Time Series Forecasting API",
    description="API for configuring, training, and predicting time series models",
    version="1.0.0"
)

app.include_router(router, prefix="/api")

