from labml import monit
from labml_app.db import init_mongo_db, computer, user
from labml_app.analyses.experiments.stdout import update_stdout, StdOutModel, StdOutIndex
from labml_app.analyses.experiments.stderr import StdErrModel, StdErrIndex, update_stderr
from labml_app.analyses.experiments.stdlogger import StdLoggerModel, StdLoggerIndex, update_std_logger
from labml_app.analyses.logs import LogPageModel


def fix_computer():
    computer_keys = computer.Computer.get_all()
    for computer_key in monit.iterate('computer_keys', computer_keys):
        c = computer_key.read()
        if 'pending_jobs' in c:
            c.pop('pending_jobs')
        if 'completed_jobs' in c:
            c.pop('completed_jobs')

        computer_key.save(c)


def fix_stdout():
    u = user.get_by_session_token('local')
    default_project = u.default_project
    runs_list = default_project.get_runs()

    for r in monit.iterate('runs', runs_list):
        run_uuid = r.run_uuid
        stdout = r.stdout + r.stdout_unmerged
        stderr = r.stderr + r.stderr_unmerged
        stdlogger = r.logger + r.logger_unmerged

        # run only if there's no stdout, stderr or stdlogger
        stdout_key = StdOutIndex.get(run_uuid)
        if stdout_key is not None:
            continue

        if stdout:
            key = StdOutIndex.get(run_uuid)
            std_out: StdOutModel

            if key is None:
                std_out = StdOutModel()
                std_out.save()
                StdOutIndex.set(run_uuid, std_out.key)
            else:
                std_out = key.load()

            std_out.update_logs_bulk(stdout)
            std_out.save()
        if stderr:
            key = StdErrIndex.get(run_uuid)
            std_err: StdErrModel

            if key is None:
                std_err = StdErrModel()
                std_err.save()
                StdErrIndex.set(run_uuid, std_err.key)
            else:
                std_err = key.load()

            std_err.update_logs_bulk(stderr)
            std_err.save()
        if stdlogger:
            key = StdLoggerIndex.get(run_uuid)
            std_logger: StdLoggerModel

            if key is None:
                std_logger = StdLoggerModel()
                std_logger.save()
                StdLoggerIndex.set(run_uuid, std_logger.key)
            else:
                std_logger = key.load()

            std_logger.update_logs_bulk(stdlogger)
            std_logger.save()


def clear_all_logs():
    stdout_keys = StdOutModel.get_all()
    for stdout_key in monit.iterate('stdout_keys', stdout_keys):
        stdout_key.delete()
    stderr_keys = StdErrModel.get_all()
    for stderr_key in monit.iterate('stderr_keys', stderr_keys):
        stderr_key.delete()
    std_logger_keys = StdLoggerModel.get_all()
    for std_logger_key in monit.iterate('std_logger_keys', std_logger_keys):
        std_logger_key.delete()

    u = user.get_by_session_token('local')
    default_project = u.default_project
    runs_list = default_project.get_runs()

    for r in monit.iterate('runs', runs_list):
        StdOutIndex.delete(r.run_uuid)
        StdLoggerIndex.delete(r.run_uuid)
        StdErrIndex.delete(r.run_uuid)

    log_keys = LogPageModel.get_all()
    for log_key in monit.iterate('log_keys', log_keys):
        log_key.delete()


def archive_all_runs():
    u = user.get_by_session_token('local')
    default_project = u.default_project
    runs_list = default_project.get_runs("all")

    for r in monit.iterate('runs', runs_list):
        if r.parent_folder == '':
            default_project.add_to_folder('archive', r)

    default_project.save()


if __name__ == '__main__':
    init_mongo_db()

    archive_all_runs()
