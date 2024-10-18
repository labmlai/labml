from labml_app.db import init_mongo_db, run

init_mongo_db()

uuid = 'c48ac8bc8ccb11efb77ba088c26a9b7a'
r = run.get(uuid)
print(r.status.load().run_status.load())