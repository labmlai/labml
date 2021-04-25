import time

from labml import logger
from labml.internal.computer.projects.methods import METHODS
from labml.logger import Text


class Polling:
    def __init__(self):
        from labml.internal.computer.projects.api import DirectApiCaller
        from labml.internal.computer.configs import computer_singleton

        self.caller = DirectApiCaller(computer_singleton().web_api_polling,
                                      {'computer_uuid': computer_singleton().uuid},
                                      timeout_seconds=15)
        self.results = []

    def run(self):
        retries = 1
        while True:
            response = self.caller.send({'jobs': self.results})
            if response is None:
                logger.log(f'Retrying again in 10 seconds ({retries})...', Text.highlight)
                time.sleep(10)
                retries += 1
                continue
            retries = 1
            self.results = []
            jobs = response.get('jobs', [])
            print(jobs)
            for j in jobs:
                self.do_job(j)

    def do_job(self, job):
        method = job['method']
        data = job['data']
        uuid = job['uuid']

        result = {
            'uuid': uuid,
            'status': 'success',
            'data': METHODS[method](**data)
        }

        self.results.append(result)


def _test():
    polling = Polling()
    polling.run()


if __name__ == '__main__':
    _test()
