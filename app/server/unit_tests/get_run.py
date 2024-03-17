from labml_app.db import run, init_mongo_db, status
from labml_app.settings import FLOAT_PROJECT_TOKEN

init_mongo_db()
run_uuid = 'b07f401ee44c11ee8a78c4cbe1b5376c'

r = run.get('b07f401ee44c11ee8a78c4cbe1b5376c')
print(r)

r = run.get(run_uuid)
if r is not None:
    uuids = [f'{run_uuid}_{i}' for i in range(1, r.world_size)]
    uuids.append(run_uuid)
    status_data = run.get_merged_status_data(uuids)

print(status_data)
