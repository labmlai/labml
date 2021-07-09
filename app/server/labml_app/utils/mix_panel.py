import queue
import inspect
import re
import threading
import time
from functools import wraps
from typing import NamedTuple, Dict, Union, Callable

from fastapi import Request

from . import slack

try:
    import mixpanel
except ImportError:
    mixpanel = None

from .. import auth

QUEUE = queue.Queue()


class EnqueueingConsumer(object):
    @staticmethod
    def send(endpoint, json_message, api_key=None):
        QUEUE.put([endpoint, json_message])


class Event:
    def __init__(self):
        if mixpanel:
            self.__mp = mixpanel.Mixpanel("7e19de9c3c68ba5a897f19837042a826", EnqueueingConsumer())
        else:
            self.__mp = None

    def _track(self, identifier: str, event: str, data: Dict) -> None:
        if self.__mp:
            self.__mp.track(identifier, event, data)

    def people_set(self, identifier: str, first_name: str, last_name: str, email: str) -> None:
        if self.__mp:
            self.__mp.people_set(identifier, {
                '$first_name': first_name,
                '$last_name': last_name,
                '$email': email,
            })

    def run_claimed_set(self, identifier: str) -> None:
        if self.__mp:
            self.__mp.people_set(identifier, {
                '$has_run_claimed': True
            })

    def computer_claimed_set(self, identifier: str) -> None:
        if self.__mp:
            self.__mp.people_set(identifier, {
                '$has_computer_claimed': True
            })

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

        remote_addr = request.client.host

        meta = {'remote_ip': remote_addr,
                'uuid': uuid,
                'labml_token': request.query_params.get('labml_token', ''),
                'labml_version': request.query_params.get('labml_version', ''),
                'agent': request.headers['User-Agent']
                }

        return meta

    def track(self, request: Request, event: str, data: Union[NamedTuple, Dict], identifier: str = '') -> None:
        if isinstance(data, NamedTuple):
            data = dict(data)

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

                if time_limit and total_time > time_limit + 1.5:
                    slack.client.send(f'PERF time: {total_time * 1000:.2f}ms method:{func.__name__}')

                self.track(request, func.__name__, {'time_elapsed': str(total_time)})

                return r

            return time_wrapper

        return decorator_function


class MixPanelThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        if mixpanel:
            self.__consumer = mixpanel.Consumer()
        else:
            self.__consumer = None

    def run(self) -> None:
        while True:
            job = QUEUE.get()
            if self.__consumer:
                self.__consumer.send(*job)

            time.sleep(5)


MixPanelEvent = Event()
