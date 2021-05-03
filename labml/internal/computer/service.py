import os
import sys
from typing import Optional

from labml import logger, monit
from labml.logger import Text


class Service:
    def __init__(self):
        from labml.internal.computer.configs import computer_singleton
        self.home = computer_singleton().home
        self.service_path = computer_singleton().home / '.config' / 'systemd' / 'user' / 'labml.service'

    def _get_service_file_content(self):
        argv = sys.argv
        # assert len(argv) == 2
        # assert argv[1] == 'monitor'
        lines = [
            '[Unit]',
            'Description=labml.ai monitoring service',
            'After=network.target',
            '',
            '[Service]',
            'Type=simple',
            f'ExecStart={argv[0]} service-run',
            'Restart=always',
            '',
            '[Install]',
            'WantedBy=default.target',
        ]

        return '\n'.join(lines)

    def _create(self):
        if self.service_path.exists():
            logger.log(f'{self.service_path} exists')
            return

        if not self.service_path.parent.exists():
            self.service_path.parent.mkdir(parents=True)

        with open(str(self.service_path), 'w') as f:
            f.write(self._get_service_file_content())

        with monit.section('Reload systemd daemon'):
            ret = os.system('systemctl --user daemon-reload')
            if ret != 0:
                monit.fail()

        with monit.section('Enable service'):
            ret = os.system('systemctl --user enable labml.service')
            if ret != 0:
                monit.fail()

    def create(self):
        self._create()

        with monit.section(f'Create {self.service_path}'):
            logger.log(['Start the monitoring service with:\n',
                        ('systemctl --user start labml.service', Text.value),
                        '\n, stop with:\n',
                        ('systemctl --user stop labml.service', Text.value),
                        '\n and check status with:\n',
                        ('systemctl --user status labml.service', Text.value),
                        ])

        with monit.section('Starting service'):
            ret = os.system('systemctl --user start labml.service')
            if ret != 0:
                monit.fail()

    def set_token(self):
        from labml.internal.computer.configs import computer_singleton

        if not computer_singleton().web_api.is_default:
            return True

        while True:
            token = input('Enter app.labml.ai token (Go to Settings after logging into app.labml.ai):')

            if len(token) != 32:
                logger.log("Invalid token", Text.danger)

            break

        computer_singleton().set_token(token)


_internal: Optional[Service] = None


def service_singleton() -> Service:
    global _internal
    if _internal is None:
        _internal = Service()

    return _internal


def _test():
    s = Service()
    print(s)
    # s.create()
    s.set_token()


if __name__ == '__main__':
    _test()
