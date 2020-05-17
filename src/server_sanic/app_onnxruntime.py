import os
from sanic import Sanic
from sanic.request import Request
from sanic.response import json

from src.dnn.onnxruntime.predictor import PredictorResNet50, PredictorMLP

app = Sanic(name='pytorch')

model_name = os.environ.get('MODEL', 'resnet50')
if model_name == 'resnet50':
    predictor = PredictorResNet50()
else:
    predictor = PredictorMLP()


@app.route('/', methods=['GET'])
async def root(request: Request):
    return json({'message': 'hello'}, status=200)


@app.route('/predict', methods=['GET'])
async def predict(request: Request):
    try:
        result = predictor.predict()
        return json(result, status=200)
    except Exception as e:
        return json({'error': str(e)}, status=500)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    workers = int(os.environ.get('NUM_WORKERS', 1))
    app.run(host='0.0.0.0', port=port, processes=workers)
