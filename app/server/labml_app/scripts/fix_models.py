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

        if stdout:
            stdout_list = stdout.split('\n')
            for i in range(len(stdout_list) - 1):
                stdout_list[i] += '\n'
            for line in stdout_list:
                update_stdout(run_uuid, line)
        if stderr:
            stderr_list = stderr.split('\n')
            for i in range(len(stderr_list) - 1):
                stderr_list[i] += '\n'
            for line in stderr_list:
                update_stderr(run_uuid, line)
        if stdlogger:
            stdlogger_list = stdlogger.split('\n')
            for i in range(len(stdlogger_list) - 1):
                stdlogger_list[i] += '\n'
            for line in stdlogger_list:
                update_std_logger(run_uuid, line)


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


if __name__ == '__main__':
    init_mongo_db()

    fix_stdout()
