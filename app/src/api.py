from fastapi import FastAPI, Request, Response
import uvicorn
from agent.train import train_model
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np

#class train_response(BaseModel):
#    message: DataFrame

app = FastAPI()

@app.post("/train")
async def train(request: Request):
    data = await request.json()
    print(data)
    #resultado = train_model()
    return StreamingResponse(train_model(data['model_name'], data['episodes'], data['days']), media_type="application/json")


@app.get("/output")
async def output(model_name):
    file_path = f"serving\\outputs\\{model_name}_output.csv"
    with open(file_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    return Response(content=csv_content, media_type="text/csv")

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/models")
async def read_models():
    models_path = "\\models"
    models = os.listdir(models_path)
    return {"models": models}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
