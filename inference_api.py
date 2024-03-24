from fastapi import FastAPI
import tensorflow as tf
import sys
# sys.path.append('..')
from src import model_inference
import pickle
import os


app = FastAPI()


@app.post("/predict")
async def root(data_path: str="rushikesh"):

    with open(os.path.join(data_path), "rb") as f:
        data = pickle.load(f)

    infer = model_inference.MnistInfer(data)
    infer.inference_data_prep(data)
    infer.load_model(model_path="models/cnn_mymodel.h5")
    predictions = infer.predictions()


    return {"predicted class": f"{predictions}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)
