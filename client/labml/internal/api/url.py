import webbrowser

from labml import logger
from labml.internal.api import ApiResponseHandler
from labml.logger import Text


class ApiUrlHandler(ApiResponseHandler):
    def __init__(self, open_browser: bool, label: str):
        self.label = label
        self.open_browser = open_browser
        self.notified = False

    def handle(self, data) -> bool:
        if self.notified:
            return False

        if 'url' not in data:
            return False

        url = data['url']
        logger.log([(self.label, Text.meta), (url, Text.link)])
        if self.open_browser:
            webbrowser.open(url)

        self.notified = True

        return False
