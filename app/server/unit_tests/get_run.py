from labml_app.db import run, init_mongo_db, status
from labml_app.db import dist_run
from labml_app.settings import FLOAT_PROJECT_TOKEN

init_mongo_db()
run_uuid = 'a9fff63415e711f094de00001047fe80'

r = dist_run.get_main_run(run_uuid)
r.comment = 'change here'
r.save()