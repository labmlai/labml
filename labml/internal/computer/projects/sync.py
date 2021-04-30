from labml.internal.computer.projects import Projects


def sync():
    projects = Projects()
    runs = projects.get_runs()

    from labml.internal.computer.projects.api import DirectApiCaller
    from labml.internal.computer.configs import computer_singleton

    sync_caller = DirectApiCaller(computer_singleton().web_api_sync,
                                  {'computer_uuid': computer_singleton().uuid},
                                  timeout_seconds=15)

    from labml import logger
    from labml.logger import inspect, Text

    response = sync_caller.send({'runs': [r.to_dict() for r in runs]})
    for k, v in response.items():
        logger.log(k, Text.title)
        inspect(v)

    # TODO: Delete runs


if __name__ == '__main__':
    sync()
