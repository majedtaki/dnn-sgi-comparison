import requests


class ProxyPyTorch:
    def __init__(self, host: str = 'localhost', port: int = 8500):
        self._endpoint = 'http://' + host + ':' + str(port) + '/predict'

    def predict(self):
        response = requests.get(self._endpoint, headers={'X-Token': 'test'})
        return response.json()
