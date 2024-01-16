from labml_app.db import run, init_mongo_db, project

init_mongo_db()

projects = project.Project.get_all()
for p in projects:
    p = p.load()
    print(p.labml_token)
