from labml_app.db import run, init_db, project

init_db()

run_keys = run.Run.get_all()
for run_key in run_keys:
    r = run_key.load()
    if r.is_claimed and not r.owner:
        run.delete(r.run_uuid)

print(len(run_keys))
