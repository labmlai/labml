from labml_app.db import run, init_db

init_db()

r = run.get('18cfffe437f711ee8e4254ef33f17c7c')

print(r)
