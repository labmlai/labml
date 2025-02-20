from datetime import datetime

from labml_app.db import init_mongo_db, run

init_mongo_db()

uuid = '67a33d2ed1e911ef8144a088c2998ee2'
r = run.get(uuid)
print(r)
print(r.status.load().run_status.load())

