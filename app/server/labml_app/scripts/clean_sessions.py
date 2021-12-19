from labml import monit
from labml_app.db import project, init_db

init_db()

project_keys = project.Project.get_all()
for project_key in monit.iterate('Projects', project_keys):
    p: project.Project = project_key.load()
    p.sessions = {}
    p.save()
