try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError as e:
    WebClient = None
    SlackApiError = None

from labml_app import settings


class SlackMessage:
    def __init__(self):
        self._client = WebClient(settings.SLACK_BOT_TOKEN)

    def send(self, text):
        res = {'error': '', 'success': False, 'ts': ''}

        if settings.SLACK_BOT_TOKEN and settings.SLACK_CHANNEL:
            try:
                ret = self._client.chat_postMessage(
                    channel=settings.SLACK_CHANNEL,
                    text=text,
                )
                res['ts'] = ret['ts']
                res['success'] = True
            except SlackApiError as e:
                res['error'] = e.response["error"]

        return res


class DummySlackMessage:
    def send(self, text):
        return {'error': '', 'success': False, 'ts': ''}


if WebClient is not None:
    client = SlackMessage()
else:
    client = DummySlackMessage()
