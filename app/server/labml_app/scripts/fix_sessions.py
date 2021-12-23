from labml import monit
from labml_app.db import computer, init_db

res = {}

init_db()
computer_keys = computer.Computer.get_all()
for computer_key in monit.iterate('computers', computer_keys):
    c: computer.Computer = computer_key.load()

    res[type(c.sessions)] = res.get(type(c.sessions), 0)
    res[type(c.sessions)] += 1
print(res)