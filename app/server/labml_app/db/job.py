import time
from typing import Optional, Dict, Union, Any

from labml_db import Model, Index

from labml_app import utils


class JobStatuses:
    INITIATED = 'initiated'
    FAIL = 'fail'
    SUCCESS = 'success'
    TIMEOUT = 'timeout'
    COMPUTER_OFFLINE = 'computer_offline'


class JobMethods:
    START_TENSORBOARD = 'start_tensorboard'
    DELETE_RUNS = 'delete_runs'
    CLEAR_CHECKPOINTS = 'clear_checkpoints'
    CALL_SYNC = 'call_sync'


NON_REPEATED_METHODS = [JobMethods.CALL_SYNC]

JobDict = Dict[str, Union[str, float]]


class Job(Model['Job']):
    job_uuid: str
    method: str
    status: str
    created_time: float
    completed_time: float
    data: Dict[str, Any]

    @classmethod
    def defaults(cls):
        return dict(job_uuid='',
                    method='',
                    status='',
                    created_time=None,
                    completed_time=None,
                    data={},
                    )

    @property
    def is_success(self) -> bool:
        return self.status == JobStatuses.SUCCESS

    @property
    def is_error(self) -> bool:
        return self.status == JobStatuses.FAIL

    @property
    def is_completed(self) -> bool:
        return self.status == JobStatuses.FAIL or self.status == JobStatuses.SUCCESS

    @property
    def is_non_repeated(self) -> bool:
        return self.method in NON_REPEATED_METHODS

    def to_data(self) -> JobDict:
        return {
            'uuid': self.job_uuid,
            'method': self.method,
            'status': self.status,
            'created_time': self.created_time,
            'completed_time': self.completed_time,
            'data': self.data
        }

    def update_job(self, status: str, data: Dict[str, Any]) -> None:
        self.status = status

        if self.status in [JobStatuses.SUCCESS, JobStatuses.FAIL]:
            self.completed_time = time.time()

        if type(data) is dict:
            self.data.update(data)

        self.save()


class JobIndex(Index['Job']):
    pass


def create(method: str, data: Dict[str, Any]) -> Job:
    job = Job(job_uuid=utils.gen_token(),
              method=method,
              created_time=time.time(),
              data=data,
              status=JobStatuses.INITIATED,
              )
    job.save()
    JobIndex.set(job.job_uuid, job.key)

    return job


def get(job_uuid: str) -> Optional[Job]:
    job_key = JobIndex.get(job_uuid)

    if job_key:
        return job_key.load()

    return None


def delete(job_uuid: str) -> None:
    job_key = JobIndex.get(job_uuid)

    if job_key:
        job_key.delete()
        JobIndex.delete(job_uuid)
