from labml_remote.job import JOBS
from labml_remote.server import SERVERS

server = next(iter(SERVERS))
JOBS.create(server, 'python hello_world_sleep.py', {}, ['hello', 'master']).start()
JOBS.create(server, 'python hello_world_sleep.py', {}, ['hello']).start()
