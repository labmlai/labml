import threading
import time
import webbrowser
from typing import Dict, Optional, TYPE_CHECKING

from labml import logger
from labml.internal.api import ApiCaller, Packet, ApiDataSource
from labml.logger import Text
from ..configs.processor import ConfigsSaver

if TYPE_CHECKING:
    from ..experiment.experiment_run import Run

LOGS_FREQUENCY = 0


class WebApiConfigsSaver(ConfigsSaver):
    def __init__(self, api_experiment: 'ApiExperiment'):
        self.api_experiment = api_experiment

    def save(self, configs: Dict):
        self.api_experiment.save_configs(configs)


class ApiExperiment(ApiDataSource):
    configs_saver: Optional[WebApiConfigsSaver]

    def __init__(self, api_caller: ApiCaller, *,
                 frequency: float,
                 open_browser: bool):
        super().__init__()

        self.frequency = frequency
        self.open_browser = open_browser
        self.api_caller = api_caller
        self.configs_saver = None
        self.data = {}
        self.callback = None
        self.lock = threading.Lock()

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = WebApiConfigsSaver(self)
        return self.configs_saver

    def save_configs(self, configs: Dict[str, any]):
        with self.lock:
            self.data['configs'] = configs

        self.api_caller.has_data(self)

    def get_data_packet(self) -> Packet:
        with self.lock:
            self.data['time'] = time.time()
            packet = Packet(self.data, callback=self.callback)
            self.data = {}
            self.callback = None
            return packet

    def start(self, run: 'Run'):
        with self.lock:
            self.data.update(dict(
                name=run.name,
                comment=run.comment,
                python_file=run.python_file,
                repo_remotes=run.repo_remotes,
                commit=run.commit,
                commit_message=run.commit_message,
                is_dirty=run.is_dirty,
                start_step=run.start_step,
                load_run=run.load_run,
                tags=run.tags,
                notes=run.notes
            ))
            self.callback = self._started

        self.api_caller.has_data(self)

        from labml.internal.api.logs import API_LOGS
        API_LOGS.set_api(self.api_caller, frequency=LOGS_FREQUENCY)

    def _started(self, url):
        if url is None:
            return None

        logger.log([('Monitor experiment at ', Text.meta), (url, Text.link)])
        if self.open_browser:
            webbrowser.open(url)

    def status(self, rank: int, status: str, details: str, time_: float):
        with self.lock:
            self.data['status'] = {
                'rank': rank,
                'status': status,
                'details': details,
                'time': time_
            }

        self.api_caller.has_data(self)

        # TODO: Will have to fix this when there are other statuses that dont stop the experiment
        # This will stop the thread after sending all the data
        self.api_caller.stop()
