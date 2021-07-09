import time

from labml import logger
from labml.internal.computer.projects.methods import METHODS
from labml.logger import Text, inspect


class Polling:
    def __init__(self):
        from labml.internal.computer.projects.api import DirectApiCaller
        from labml.internal.computer.configs import computer_singleton

        self.caller = DirectApiCaller(computer_singleton().web_api_polling,
                                      {'computer_uuid': computer_singleton().uuid},
                                      timeout_seconds=60)
        self.results = []
        self.is_stopped = False

    def run(self):
        retries = 1
        while not self.is_stopped:
            response = self.caller.send({'jobs': self.results})
            if response is None:
                logger.log(f'Retrying again in 10 seconds ({retries})...', Text.highlight)
                time.sleep(10)
                retries += 1
                continue
            retries = 1
            self.results = []
            jobs = response.get('jobs', [])
            logger.log(f'Jobs: {len(jobs)}')
            for j in jobs:
                inspect(j)
                res = self.do_job(j)
                self.results.append(res)
                inspect(res)

    def do_job(self, job):
        method = job['method']
        data = job['data']
        uuid = job['uuid']

        status, data = METHODS[method](**data)
        result = {
            'uuid': uuid,
            'status': status,
            'data': data,
        }

        return result


def _test():
    polling = Polling()
    polling.run()


if __name__ == '__main__':
    _test()
