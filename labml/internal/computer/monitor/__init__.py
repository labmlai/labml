from labml.internal.api import ApiCaller
from labml.internal.computer.configs import computer_singleton
from labml.internal.computer.monitor.process import ProcessMonitor
from labml.internal.computer.monitor.scanner import Scanner
from labml.internal.computer.writer import Writer, Header


class MonitorComputer:
    def __init__(self, session_uuid: str, open_browser):
        api_caller = ApiCaller(computer_singleton().web_api.url,
                               {'computer_uuid': computer_singleton().uuid, 'session_uuid': session_uuid},
                               timeout_seconds=120,
                               daemon=True)
        self.writer = Writer(api_caller, frequency=computer_singleton().web_api.frequency)
        self.header = Header(api_caller,
                             frequency=computer_singleton().web_api.frequency,
                             open_browser=open_browser)
        self.scanner = Scanner()

    def start(self):
        self.header.start(self.scanner.configs())
        self.writer.track(self.scanner.first())

    def track(self):
        self.writer.track(self.scanner.track())
