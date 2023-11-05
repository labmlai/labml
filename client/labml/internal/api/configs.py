class WebAPIConfigs:
    url: str
    frequency: float
    verify_connection: bool
    open_browser: bool

    def __init__(self, *,
                 url: str,
                 frequency: float,
                 verify_connection: bool,
                 open_browser: bool,
                 is_default: bool):
        self.is_default = is_default
        self.open_browser = open_browser
        self.frequency = frequency
        self.verify_connection = verify_connection
        self.url = url
