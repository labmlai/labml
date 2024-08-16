import json

import requests


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

        # return response
        try:
            return response.json()
        except json.JSONDecodeError:
            raise NetworkError(response.status_code, url, 'JSON decode error', response.text)


class AppAPI:
    def __init__(self, base_url="http://localhost:5005/api/v1"):
        self.network = Network(base_url)

    def get_run(self, run_uuid):
        return self.network.send_http_request('GET', f'/run/{run_uuid}')

    """
    Updates the run data
    data: {
        'name': str,
        'comment': str,
        'note': str,
        'favourite_configs': List[str],
        'selected_configs': List[str],
        'tags': List[str],
    }
    """

    def update_run_data(self, run_uuid, data):
        return self.network.send_http_request('POST', f'/run/{run_uuid}', data)

    def get_run_status(self, run_uuid):
        return self.network.send_http_request('GET', f'/run/status/{run_uuid}')

    def get_runs(self):
        return self.network.send_http_request('GET', f'/runs/null')

    def get_runs_by_tag(self, tag):
        return self.network.send_http_request('GET', f'/runs/null/{tag}')

    def archive_runs(self, run_uuids):
        return self.network.send_http_request('POST', '/runs/archive', {'run_uuids': run_uuids})

    def unarchive_runs(self, run_uuids):
        return self.network.send_http_request('POST', '/runs/unarchive', {'run_uuids': run_uuids})

    def delete_runs(self, run_uuids):
        return self.network.send_http_request('PUT', '/runs', {'run_uuids': run_uuids})

    """
    Get analysis data
    
    url: str [compare/metrics, std_logger, stderr, metrics, stdout, battery, cpu, memory, disk, gpu, process]
    
    'get_all': bool, Either get all indicators or only the selected indicators
    'current_uuid': str, Current run uuid for comparisons (only required for comparison metrics)
    """

    def get_analysis(self, url: str, run_uuid: str, *,
                     get_all=False,
                     current_uuid: str = ''):
        method = 'GET'

        if url == 'compare/metrics' or url == 'metrics':
            method = 'POST'

        return self.network.send_http_request(method, f"/{url}/{run_uuid}?current={current_uuid}",
                                              {'get_all': get_all})

    """
        Get analysis preferences

        url: str [compare/metrics, std_logger, stderr, metrics, stdout, battery, cpu, memory, disk, gpu, process]
    """

    def get_preferences(self, url: str, run_uuid):
        return self.network.send_http_request('GET', f'{url}/preferences/{run_uuid}')

    """
        Update analysis preferences

        url: str [compare/metrics, std_logger, stderr, metrics, stdout, battery, cpu, memory, disk, gpu, process]
        data: {
            'series_preferences': List[int],  # Preferences for series data
            'series_names': List[str],  # Names of the series indicators
            'chart_type': int,  # Type of chart to display
            'step_range': List[int],  # Range of steps to display. Two integers. -1 for no bound
            'focus_smoothed': bool,  # Whether to focus on smoothed data
            'smooth_value': float,  # Value for smoothing the data. in (0, 1)
            'smooth_function': str,  # Function used for smoothing [exponential, left_exponential]
            # following needed for comparisons
            'base_experiment': str,  # Base experiment for
            'base_series_preferences': List[int],  # Preferences for base series data
            'base_series_names': List[str],  # Names of the base series indicators
        }
        
        - series_preferences content:
        -1 -> not selected
        1 -> selected
    """

    def update_preferences(self, url: str, run_uuid, data):
        return self.network.send_http_request('POST', f'{url}/preferences/{run_uuid}', data)

    """
    data {
        'name': str,
        'description': str
    }
    """

    def create_custom_metric(self, run_uuid, data):
        return self.network.send_http_request('POST', f'/custom_metrics/{run_uuid}/create', data)

    def get_custom_metrics(self, run_uuid):
        return self.network.send_http_request('GET', f'/custom_metrics/{run_uuid}')

    """
    data {
        id: str,
        preferences: dict # same as set preference method data
        name: str,
        description: str
    }
    """

    def update_custom_metric(self, run_uuid, data):
        return self.network.send_http_request('POST', f'/custom_metrics/{run_uuid}', data)

    def delete_custom_metric(self, run_uuid, metric_id):
        return self.network.send_http_request('POST', f'/custom_metrics/{run_uuid}/delete',
                                              {'id': metric_id})

    """
    url: str [stderr, std_logger, stdout]
    page_no: int 
    -1 -> get all pages
    -2 -> get last page
    i -> get ith page
    """

    def get_logs(self, run_uuid: str, url: str, page_no: int):
        return self.network.send_http_request('POST', f'/logs/{url}/{run_uuid}', {'page': page_no})
