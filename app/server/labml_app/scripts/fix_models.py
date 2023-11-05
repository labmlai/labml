from labml import monit

from labml_app.db import init_mongo_db, computer


def fix_computer():
    computer_keys = computer.Computer.get_all()
    for computer_key in monit.iterate('computer_keys', computer_keys):
        c = computer_key.read()
        if 'pending_jobs' in c:
            c.pop('pending_jobs')
        if 'completed_jobs' in c:
            c.pop('completed_jobs')

        computer_key.save(c)


if __name__ == '__main__':
    init_mongo_db()

    fix_computer()
