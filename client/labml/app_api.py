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
        self.session_token = None
        self.app_token = None

    def get_app_token(self):  # todo where to save this token
        return self.app_token

    def set_app_token(self, token):
        self.app_token = token

    def update_session(self, token):
        self.session_token = token

    def get_user(self):
        headers = {}
        app_token = self.get_app_token()
        if app_token:
            headers['Authorization'] = f'Bearer {app_token}'

        device_info = {
            'userAgent': platform.platform(),
            'platform': platform.system(),
            'appName': 'Python',
            'appCodeName': 'Python',
            'engine': platform.python_implementation(),
            'appVersion': platform.python_version(),
            'height': 0,
            'width': 0
        }

        data = {
            'device': device_info,
            'referrer': ''
        }

        response = requests.post(f'{self.base_url}/auth/user', json=data, headers=headers)
        res = response.json()

        if res and res.get('user') and res['user'].get('token'):
            self.set_app_token(res['user']['token'])

        return res

    def send_http_request(self, method, url, data=None, retry_auth=True):
        headers = {}
        app_token = self.session_token

        if app_token:
            headers['Authorization'] = f'Bearer {app_token}'

        if data:
            headers['Content-Type'] = 'application/json'

        full_url = self.base_url + url
        response = requests.request(method, full_url, json=data, headers=headers)

        if 'Authorization' in response.headers:
            token = response.headers['Authorization']
            if '/auth/sign_in' in url or '/auth/sign_up' in url:
                self.set_app_token(token)
            else:
                self.update_session(token)

        if response.status_code == 401 and retry_auth:
            self.get_user()
            return self.send_http_request(method, url, data, False)

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
