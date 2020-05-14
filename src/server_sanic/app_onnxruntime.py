import argparse
from sanic import Sanic
from sanic.request import Request
from sanic.response import json

from src.dnn.onnxruntime.predictor import Predictor

app = Sanic(name='pytorch')

predictor = Predictor()


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=8000, workers=args.workers)
