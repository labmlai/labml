import platform
import requests
import json


class NetworkError(Exception):
    def __init__(self, status_code, url, message=None, description=None):
        self.status_code = status_code
        self.url = url
        self.message = message
        self.error_description = description
        self.stack_trace = None
        try:
            json_message = json.loads(message)
            self.stack_trace = json_message.get('trace')
        except json.JSONDecodeError:
            pass

    def __str__(self):
        return (f"Status Code: {self.status_code}\n"
                f"URL: {self.url}\nDescription: {self.error_description}\n"
                f"Message: {self.message}\n"
                f"StackTrace: {self.stack_trace}")


class Network:
    def __init__(self, base_url):
        self.base_url = base_url

    def send_http_request(self, method, url, data=None):
        headers = {}

        if data:
            headers['Content-Type'] = 'application/json'

        full_url = self.base_url + url
        response = requests.request(method, full_url, json=data, headers=headers)

        if response.status_code >= 400:
            error_message = None
            if response.json():
                if 'error' in response.json():
                    error_message = response.json()['error']
                elif 'data' in response.json() and 'error' in response.json()['data']:
                    error_message = response.json()['data']['error']
            raise NetworkError(response.status_code, url, response.text, error_message)

        return response.json()


class AppAPI:
    def __init__(self, base_url="http://localhost:5005/api/v1"):
        self.network = Network(base_url)

    def get_runs(self):
        return self.network.send_http_request('GET', '/runs/null')
