import os
import uvicorn
from fastapi import FastAPI, Header, HTTPException

from src.dnn.onnxruntime.predictor import PredictorResNet50, PredictorMLP

app = FastAPI()
model_name = os.environ.get('MODEL', 'resnet50')
if model_name == 'resnet50':
    predictor = PredictorResNet50()
else:
    predictor = PredictorMLP()


@app.get('/')
async def root():
    return {'message': 'hello'}


@app.get('/predict')
async def predict(*, x_token: str = Header(None)):
    if x_token != 'test':
        raise HTTPException(status_code=403, detail={"error": "invalid header value of X-Token"})
    try:
        result = predictor.predict()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
