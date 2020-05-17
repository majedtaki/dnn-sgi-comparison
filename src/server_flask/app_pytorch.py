import os
from flask import Flask, jsonify

from src.dnn.pytorch.predictor import PredictorResNet50, PredictorMLP

app = Flask(__name__)

model_name = os.environ.get('MODEL', 'resnet50')
if model_name == 'resnet50':
    predictor = PredictorResNet50()
else:
    predictor = PredictorMLP()


@app.route('/')
def root():
    return jsonify({'message': 'hello'}), 200


@app.route('/predict')
def predict():
    try:
        prediction = predictor.predict()
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    workers = int(os.environ.get('NUM_WORKERS', 1))
    app.run(host='0.0.0.0', port=port, processes=workers)
