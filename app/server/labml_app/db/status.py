import time
from typing import Dict

from labml_db import Model, Key

from ..enums import RunEnums


class RunStatus(Model['RunStatusModel']):
    status: str
    details: object
    time: float

    @classmethod
    def defaults(cls):
        return dict(status='',
                    details=None,
                    time=None
                    )


class Status(Model['Status']):
    last_updated_time: float
    run_status: Key[RunStatus]

    @classmethod
    def defaults(cls):
        return dict(last_updated_time=None,
                    run_status=None
                    )

    # run_status can be sent as a parameter if loaded from outside
    def get_data(self, run_status: Dict[str, any] = None) -> Dict[str, any]:
        if run_status is None:
            run_status = self.run_status.load().to_dict()

        run_status['status'] = self.get_true_status(run_status.get('status', ''))

        return {
            'last_updated_time': self.last_updated_time,
            'run_status': run_status
        }

    def update_time_status(self, data: Dict[str, any]) -> None:
        self.last_updated_time = time.time()

        s = data.get('status', {})
        if s:
            run_status = self.run_status.load()

            run_status.status = s.get('status', run_status.status)
            run_status.details = s.get('details', run_status.details)
            run_status.time = s.get('time', run_status.time)

            run_status.save()

        self.save()

    def get_true_status(self, status: str = None) -> str:
        if not status:
            status = self.run_status.load().status
        not_responding = False

        if status == RunEnums.RUN_IN_PROGRESS:
            if self.last_updated_time is not None:
                time_diff = (time.time() - self.last_updated_time) / 60
                if time_diff > 15:
                    not_responding = True

        if not_responding:
            return RunEnums.RUN_NOT_RESPONDING
        elif status == '':
            return RunEnums.RUN_UNKNOWN
        else:
            return status


def create_status() -> Status:
    time_now = time.time()

    run_status = RunStatus(status=RunEnums.RUN_IN_PROGRESS,
                           time=time_now
                           )
    status = Status(last_updated_time=time_now,
                    run_status=run_status.key
                    )
    status.save()
    run_status.save()

    return status
