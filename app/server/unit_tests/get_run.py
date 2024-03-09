from labml_app.db import run, init_mongo_db, project
from labml_app.settings import FLOAT_PROJECT_TOKEN

init_mongo_db()

r = run.get('c')

print(r.run_uuid)
print(r.is_claimed)

float_project = project.get_project(labml_token=FLOAT_PROJECT_TOKEN)
print(r.run_uuid in float_project.runs)

print(project.get_run('5372ad8ab45311eeb035946dae1e1468', FLOAT_PROJECT_TOKEN).run_uuid)
