from datetime import datetime

from labml_app.db import init_mongo_db, run

format_string = "%Y-%m-%d %H:%M:%S"

init_mongo_db()

uuid = 'c48ac8bc8ccb11efb77ba088c26a9b7a_1'
# uuid2 = 'c48ac8bc8ccb11efb77ba088c26a9b7a_0'
# r = run.get(uuid2)
# print(r.status.load().run_status.load())
for i in range(1, 8):
    print(f'c48ac8bc8ccb11efb77ba088c26a9b7a_{i}')
    r = run.get(f'c48ac8bc8ccb11efb77ba088c26a9b7a_{i}')
    s = r.status.load()
    datetime_object = datetime.fromtimestamp(s.last_updated_time)
    print(datetime_object)
    rs = s.run_status.load()
    datetime_object = datetime.fromtimestamp(rs.time)
    print(rs.status, datetime_object)
