import json
import socket
import urllib.error
import urllib.request
from typing import Dict, List, Any

import labml
from labml.logger import Text
from labml.utils.notice import labml_notice


class DirectApiCaller:
    def __init__(self, url: str, params: Dict[str, str], *, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

        params.copy()
        params = params.copy()
        params['labml_version'] = labml.__version__
        params = '&'.join([f'{k}={v}' for k, v in params.items()])
        self.url = f'{url}{params}'

    def send(self, data: Any) -> Any:
        try:
            response = self._send(data)
        except urllib.error.HTTPError as e:
            labml_notice([f'Failed to send to {self.url}: ',
                          (str(e.code), Text.value),
                          '\n' + str(e.reason)])
            return None
        except urllib.error.URLError as e:
            labml_notice([f'Failed to connect to {self.url}\n',
                          str(e.reason)])
            return None
        except socket.timeout as e:
            labml_notice([f'{self.url} timeout\n',
                          str(e)])
            return None
        except ConnectionResetError as e:
            labml_notice([f'Connection reset by LabML App server {self.url}\n',
                          str(e)])
            return None

        return response

    def _send(self, data: List[Dict[str, any]]) -> Dict:
        req = urllib.request.Request(self.url)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        data_json = json.dumps(data)
        data_json = data_json.encode('utf-8')
        req.add_header('Content-Length', str(len(data)))

        response = urllib.request.urlopen(req, data_json, timeout=self.timeout_seconds)
        content = response.read().decode('utf-8')
        result = json.loads(content)

        for e in result.get('errors', []):
            if 'error' in e:
                labml_notice(['LabML App Error: ', (e['error'] + '', Text.key), '\n',
                              (e['message'], Text.value)],
                             is_danger=True)
                raise RuntimeError('LabML App Error', e)
            elif 'warning' in e:
                labml_notice(
                    ['LabML App Warning: ', (e['warning'] + ': ', Text.key), (e['message'], Text.value)])
            else:
                raise RuntimeError('Unknown error from LabML App', e)

        return result
