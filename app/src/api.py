from fastapi import FastAPI, Request, Response
import uvicorn
from agent.train import train_model
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

#Clases de validación de Pydantic
class TrainRequest(BaseModel):
    model_name: str
    episodes: int = Field(..., gt=0)  # Mayor a 0
    days: int = Field(..., ge=1, le=365)  # Entre 1 y 365
class TrainResponse(BaseModel):
    progress: int  # Porcentaje de progreso
    message: str
class ModelsResponse(BaseModel):
    models: list[str]
class DataResponse(BaseModel):
    data: list[str]


app = FastAPI()

@app.post("/train", response_model=dict)
async def train(request: TrainRequest):
    return StreamingResponse(
        train_model(request.model_name, request.episodes, request.days), 
        media_type="application/json")

@app.get("/output", response_class=Response)
async def output(model_name: str):
    file_path = f"serving\\outputs\\{model_name}_output.csv"
    with open(file_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    return Response(content=csv_content, media_type="text/csv")

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/models", response_model=ModelsResponse)
async def read_models():
    models_path = "serving\\models"
    models = os.listdir(models_path)
    return {"models": models}

@app.get("/get-data", response_model=DataResponse)
async def read_models():
    data_path = "serving\\app\\src\\data"
    #Listar solo los que terminen en csv
    data = [file for file in os.listdir(data_path) if file.endswith(".csv")]
    return {"data": data}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
