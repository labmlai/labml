from .process import ProcessMonitor
from .scanner import Scanner
from ..configs import computer_singleton
from ..writer import Writer, Header
from ...app import AppTracker


class MonitorComputer:
    def __init__(self, session_uuid: str, open_browser):
        api_caller = AppTracker(computer_singleton().app_configs.url,
                                {'computer_uuid': computer_singleton().uuid, 'session_uuid': session_uuid},
                                timeout_seconds=120,
                                daemon=True)
        self.writer = Writer(api_caller, frequency=computer_singleton().app_configs.frequency)
        self.header = Header(api_caller, open_browser=open_browser)
        self.scanner = Scanner()

    def start(self):
        self.header.start(self.scanner.configs())
        self.writer.track(self.scanner.first())

    def track(self):
        self.writer.track(self.scanner.track())
