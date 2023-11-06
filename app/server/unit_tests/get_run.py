from labml_app.db import run, init_mongo_db

init_mongo_db()

r = run.get('eeb8e34067ff11eeaf633275fba9aeb4')

r.name = 'XXXX'
r.tags = []
r.save()
