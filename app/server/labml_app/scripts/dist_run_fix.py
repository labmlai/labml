import uuid

from labml import monit
from labml_app.db import init_mongo_db, run, user, dist_run

"""
create dist run objects for each main run
get all runs per main run and set ranks inside it
set main_rank, world_size, owner from main run
set is_claimed true for all
"""
init_mongo_db()
u = user.get_by_session_token('local')
default_project = u.default_project

default_project.dist_runs = {}
default_project.dist_tag_index.clear()
print(default_project.dist_runs)
default_project.save()

print(default_project.labml_token)

runs = {}





index = 0
failed_count = 0
for run_uuid, run_key in zip(default_project.runs.keys(), default_project.runs.values()):
    index += 1
    print(index/len(default_project.runs))

    r = run_key.load()
    if r.rank != 0:
        continue

    dr = dist_run.get_or_create(uuid.uuid4().hex, default_project.labml_token)
    dr.world_size = r.world_size
    dr.is_claimed = True
    dr.owner = r.owner
    dr.main_rank = r.main_rank

    dr.ranks[0] = r.run_uuid
    for rank in r.get_rank_uuids():
        dr.ranks[rank] = r.get_rank_uuids()[rank]

    # assert dr.world_size == 0 or dr.world_size == len(dr.ranks)
    if dr.world_size != 0 and dr.world_size != len(dr.ranks):
        print('hehe')

    dr.save()

    default_project.add_dist_run_with_model(dr)

    from labml_app.analyses.experiments import custom_metrics, data_store

    # copy custom metrics
    cur_key = custom_metrics.CustomMetricsListIndex.get(r.run_uuid)
    custom_metrics.CustomMetricsListIndex.set(dr.uuid, cur_key)
    custom_metrics.CustomMetricsListIndex.set(r.run_uuid, None)

    # copy data_store
    cur_d_key = data_store.DataStoreIndex.get(r.run_uuid)
    data_store.DataStoreIndex.set(dr.uuid, cur_d_key)
    data_store.DataStoreIndex.set(r.run_uuid, None)

    # tags
    for tag in r.tags:
        if tag not in default_project.dist_tag_index:
            default_project.dist_tag_index[tag] = set()
        default_project.dist_tag_index[tag].add(dr.uuid)


default_project.save()

print("failed: ", failed_count)




