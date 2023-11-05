import threading
import time
from typing import Dict, Optional, TYPE_CHECKING

from . import AppTracker, Packet, AppTrackDataSource, AppTrackResponseHandler
from .dynamic import DynamicUpdateHandler
from .url import AppUrlResponseHandler
from ..configs.processor import ConfigsSaver

if TYPE_CHECKING:
    from ..experiment.experiment_run import Run

LOGS_FREQUENCY = 0


class AppConfigsSaver(ConfigsSaver):
    def __init__(self, app_experiment: 'AppExperiment'):
        self._app_experiment = app_experiment

    def save(self, configs: Dict):
        self._app_experiment.save_configs(configs)


class DynamicHyperParamHandler(AppTrackResponseHandler):
    def __init__(self, handler: DynamicUpdateHandler):
        self.handler = handler

    def handle(self, data) -> bool:
        if 'dynamic' not in data:
            return False

        self.handler.handle(data['dynamic'])
        return False


class AppExperiment(AppTrackDataSource):
    configs_saver: Optional[AppConfigsSaver]

    def __init__(self, app_tracker: AppTracker, *,
                 frequency: float,
                 open_browser: bool):
        super().__init__()

        self.frequency = frequency
        self.open_browser = open_browser
        self.app_tracker = app_tracker
        self.configs_saver = None
        self.data = {}
        self.lock = threading.Lock()

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = AppConfigsSaver(self)
        return self.configs_saver

    def save_configs(self, configs: Dict[str, any]):
        with self.lock:
            self.data['configs'] = configs

        self.app_tracker.has_data(self)

    def get_data_packet(self) -> Packet:
        with self.lock:
            self.data['time'] = time.time()
            packet = Packet(self.data)
            self.data = {}
            return packet

    def start(self, run: 'Run'):
        self.app_tracker.add_handler(AppUrlResponseHandler(self.open_browser, 'Monitor experiment at '))

        with self.lock:
            from labml.internal.computer.configs import computer_singleton

            computer_uuid = computer_singleton().uuid

            self.data.update(dict(
                name=run.name,
                comment=run.comment,
                computer=computer_uuid,
                python_file=run.python_file,
                repo_remotes=run.repo_remotes,
                commit=run.commit,
                commit_message=run.commit_message,
                is_dirty=run.is_dirty,
                start_step=run.start_step,
                load_run=run.load_run,
                tags=run.tags,
                notes=run.notes,
            ))

        self.app_tracker.has_data(self)

        from labml.internal.app.logs import APP_CONSOLE_LOGS
        APP_CONSOLE_LOGS.set_app_tracker(self.app_tracker, frequency=LOGS_FREQUENCY)

    def worker(self):
        from labml.internal.app.logs import APP_CONSOLE_LOGS
        APP_CONSOLE_LOGS.set_app_tracker(None, frequency=LOGS_FREQUENCY)

    def set_dynamic_handler(self, handler: DynamicUpdateHandler):
        self.app_tracker.add_handler(DynamicHyperParamHandler(handler))

    def status(self, rank: int, status: str, details: str, time_: float):
        with self.lock:
            self.data['status'] = {
                'rank': rank,
                'status': status,
                'details': details,
                'time': time_
            }

        self.app_tracker.has_data(self)

        # TODO: Will have to fix this when there are other statuses that dont stop the experiment
        # This will stop the thread after sending all the data
        self.app_tracker.stop()
