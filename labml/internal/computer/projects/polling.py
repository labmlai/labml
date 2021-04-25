class Polling:
    def __init__(self):
        from labml.internal.computer.projects.api import DirectApiCaller
        from labml.internal.computer.configs import computer_singleton

        self.caller = DirectApiCaller(computer_singleton().web_api_polling,
                                      {'computer_uuid': computer_singleton().uuid},
                                      timeout_seconds=15)

    def poll(self):
        pass
