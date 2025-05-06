import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.model_configurator import ModelConfigurator

router = APIRouter()
configurator = ModelConfigurator()


class ConfigRequest(BaseModel):
    trend_models: list[str]
    seasonal_models: list[str]


class FilePathRequest(BaseModel):
    path: str


@router.post("/configure")
def configure(request: ConfigRequest):
    try:
        configurator.set_config(request.trend_models, request.seasonal_models)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Configuration saved"}


@router.post("/train")
def train(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, parse_dates=["date"], index_col=False)
    try:
        configurator.fit_model(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Model training was successful with configs {configurator.config_names}"}


@router.post("/predict")
def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, parse_dates=["date"], index_col=False)
    try:
        prediction = configurator.predict(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    csv_buffer = io.StringIO()
    prediction.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return StreamingResponse(
        content=csv_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
