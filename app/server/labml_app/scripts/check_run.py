from labml_app.db import init_mongo_db, run

init_mongo_db()

uuid = 'c48ac8bc8ccb11efb77ba088c26a9b7a_1'
# uuid2 = 'c48ac8bc8ccb11efb77ba088c26a9b7a_0'
# r = run.get(uuid2)
# print(r.status.load().run_status.load())
for i in range(1, 7):
    print(f'c48ac8bc8ccb11efb77ba088c26a9b7a_{i}')
    r = run.get(f'c48ac8bc8ccb11efb77ba088c26a9b7a_{i}')
    print(r.status.load().run_status.load())