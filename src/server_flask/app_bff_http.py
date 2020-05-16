import os
from flask import Flask, jsonify

from src.proxy_http import ProxyHTTP

app = Flask(__name__)

predictor = ProxyHTTP(os.environ.get('BACKEND_HOST', 'localhost'), int(os.environ.get('BACKEND_PORT', 8500)))


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
