import time
import webbrowser
from typing import Dict, Optional

from labml import logger
from labml.internal.api import ApiCaller, Packet
from labml.logger import Text
from ..configs.processor import ConfigsSaver

LOGS_FREQUENCY = 0


class WebApiConfigsSaver(ConfigsSaver):
    def __init__(self, api_experiment: 'ApiExperiment'):
        self.api_experiment = api_experiment

    def save(self, configs: Dict):
        self.api_experiment.save_configs(configs)


class ApiExperiment:
    configs_saver: Optional[WebApiConfigsSaver]

    def __init__(self, api_caller: ApiCaller, *,
                 frequency: float,
                 open_browser: bool):
        super().__init__()

        self.frequency = frequency
        self.open_browser = open_browser
        self.api_caller = api_caller
        self.run_uuid = None
        self.name = None
        self.comment = None
        self.state = None
        self.configs_saver = None
        self.api_caller.add_state_attribute('configs')

    def set_info(self, *,
                 run_uuid: str,
                 name: str,
                 comment: str):
        self.run_uuid = run_uuid
        self.name = name
        self.comment = comment

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = WebApiConfigsSaver(self)
        return self.configs_saver

    def save_configs(self, configs: Dict[str, any]):
        self.api_caller.push(Packet({'configs': configs}))

    def start(self):
        data = {
            'name': self.name,
            'comment': self.comment,
            'time': time.time()
        }

        self.api_caller.push(Packet(data, callback=self._started))

        from labml.internal.api.logs import API_LOGS
        API_LOGS.set_api(self.api_caller, frequency=LOGS_FREQUENCY)

    def _started(self, url):
        if url is None:
            return None

        logger.log([('Monitor experiment at ', Text.meta), (url, Text.link)])
        if self.open_browser:
            webbrowser.open(url)

    def status(self, rank: int, status: str, details: str, time_: float):
        self.state = {
            'rank': rank,
            'status': status,
            'details': details,
            'time': time_
        }

        self.api_caller.push(Packet({
            'status': self.state,
            'time': time.time()
        }))

        # TODO: Will have to fix this when there are other statuses that dont stop the experiment
        # This will stop the thread after sending all the data
        self.api_caller.stop()
