import queue
import inspect
import requests
from requests.exceptions import RequestException
import re
import threading
import time
from time import sleep
from functools import wraps
from typing import NamedTuple, Dict, Union, Callable, List, Any

from fastapi import Request

from . import slack
from labml_app.logger import logger
from labml_app import settings

from .. import auth

QUEUE = queue.Queue()
ANALYTICS_ID = 'labml_app'

EXCLUDED_METHODS = {'polling'}


class Event:
    @staticmethod
    def _track(identifier: str, event: str, data: Dict) -> None:
        if settings.ANALYTICS_TO_SERVER:
            QUEUE.put({'identifier': identifier, 'event': event, 'data': data})

    def people_set(self, identifier: str, first_name: str, last_name: str, email: str) -> None:
        self._track(identifier, 'people', {'first_name': first_name, 'last_name': last_name, 'email': email})

    def run_claimed_set(self, identifier: str) -> None:
        self._track(identifier, 'run_claimed', {})

    def computer_claimed_set(self, identifier: str) -> None:
        self._track(identifier, 'computer_claimed', {})

    @staticmethod
    def has_numbers(input_string) -> bool:
        return bool(re.search(r'\d', input_string))

    def get_meta_data(self, request: Request) -> Dict[str, str]:
        run_uuid = request.query_params.get('run_uuid', '')
        computer_uuid = request.query_params.get('computer_uuid', '')

        uuid = ''
        if run_uuid:
            uuid = run_uuid
        elif computer_uuid:
            uuid = computer_uuid
        else:
            value = request.base_url.path.split('/')[-1]
            if self.has_numbers(value):
                uuid = value

        ip_address = request.headers.get('CF-Connecting-IP', request.client.host)
        user_agent = request.headers.get('User-Agent', '')

        meta = {'ip_address': ip_address,
                'time': time.time(),
                'uuid': uuid,
                'labml_token': request.query_params.get('labml_token', ''),
                'labml_version': request.query_params.get('labml_version', ''),
                'user_agent': user_agent
                }

        return meta

    def track(self, request: Request, event: str, data: Dict, identifier: str = '') -> None:

        assert isinstance(data, dict)

        user = auth.get_auth_user(request)
        if user:
            identifier = user.email

        data.update(self.get_meta_data(request))

        return self._track(identifier, event, data)

    def time_this(self, time_limit: float = None) -> Callable:
        def decorator_function(func):
            @wraps(func)
            async def time_wrapper(request: Request, *args, **kwargs):
                start = time.time()

                if inspect.iscoroutinefunction(func):
                    r = await func(request, *args, **kwargs)
                else:
                    r = func(request, *args, **kwargs)

                end = time.time()

                total_time = end - start
                if time_limit and total_time < time_limit:
                    return r

                if time_limit and total_time > time_limit + 1.5 and func.__name__ not in EXCLUDED_METHODS:
                    slack.client.send(f'PERF time: {total_time * 1000:.2f}ms method:{func.__name__}, url:{request.url}')

                self.track(request, func.__name__, {'time_elapsed': str(total_time)})

                return r

            return time_wrapper

        return decorator_function


class AnalyticsWriterThread(threading.Thread):
    """
    Writes the queue to file
    """
    file_path = str

    def __init__(self):
        self.analytics_url = 'https://analytics.papers.bar/api/v1/track'
        super().__init__(daemon=True)

    def _write_to_server(self, to_insert: List[Dict[str, Any]]) -> None:
        try:
            res = requests.post(url=f'{self.analytics_url}/{ANALYTICS_ID}', json={'track': to_insert})

            attempts = 0
            while res.status_code != 200 and attempts < 5:
                logger.error(f'error: {res.status_code}; client id {ANALYTICS_ID}, retrying...{attempts}')

                sleep(3)

                res = requests.post(url=f'{self.analytics_url}/{ANALYTICS_ID}', json={'track': to_insert})
                attempts += 1
        except RequestException as e:
            logger.error(f'error: {str(e)}; client id {ANALYTICS_ID}')

    def run(self) -> None:
        logger.info('starting AnalyticsWriter Thread')
        while True:
            jobs = []
            qsize = QUEUE.qsize()
            for i in range(min(qsize, 25)):
                job = QUEUE.get()
                jobs.append(job)

            if jobs:
                QUEUE.task_done()
                self._write_to_server(jobs)

            time.sleep(5)


analytics_writer_tread = AnalyticsWriterThread()
analytics_writer_tread.start()
AnalyticsEvent = Event()
