import uuid

from labml_db import load_keys

from labml import monit
from labml_app.db import init_mongo_db, run, user, dist_run
from labml_app.db.run import RunIndex

"""
create dist run objects for each main run
get all runs per main run and set ranks inside it
set main_rank, world_size, owner from main run
set is_claimed true for all
"""
init_mongo_db()
u = user.get_by_session_token('local')

run_to_parent = {}


def update_runs():
    default_project = u.default_project

    default_project.dist_runs = {}
    default_project.dist_tag_index = {}
    default_project.save()

    failed_count = 0
    for run_uuid, run_key in monit.iterate(default_project.runs.items()):

        r = run_key.load()
        if r is None:
            failed_count += 1
            continue
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
            rank_run = RunIndex.get(dr.ranks[rank]).load()
            if rank_run is not None:
                rank_run.parent_run_uuid = dr.uuid
                run_to_parent[rank_run.run_uuid] = dr.uuid
                rank_run.save()

        dr.save()

        default_project.add_dist_run_with_model(dr)

        from labml_app.analyses.experiments import custom_metrics, data_store

        # copy custom metrics
        cur_key = custom_metrics.CustomMetricsListIndex.get(r.run_uuid)
        if cur_key is not None:
            custom_metrics.CustomMetricsListIndex.set(dr.uuid, cur_key)
            custom_metrics.CustomMetricsListIndex.delete(r.run_uuid)

        # copy data_store
        cur_d_key = data_store.DataStoreIndex.get(r.run_uuid)
        if cur_d_key is not None:
            data_store.DataStoreIndex.set(dr.uuid, cur_d_key)
            data_store.DataStoreIndex.delete(r.run_uuid)
    default_project.save()

    print("failed: ", failed_count)


def update_preferences():
    # update preferences
    from labml_app.analyses.experiments.custom_metrics import CustomMetricsListIndex
    from labml_app.analyses import preferences
    custom_metric_list = [c.metrics for c in load_keys(CustomMetricsListIndex.get_all()) if c is not None]
    custom_metric_list = [item for sublist in custom_metric_list for item in sublist]
    custom_metric_keys = [c[1] for c in custom_metric_list if c is not None]
    custom_metric_models = [c.load() for c in custom_metric_keys]
    pref_keys = [c.preference_key for c in custom_metric_models if c is not None]

    preferences = load_keys(pref_keys)
    for pref in monit.iterate(preferences):
        if pref is None:
            continue
        if pref.base_experiment != "" and pref.base_experiment in run_to_parent:
            pref.base_experiment = run_to_parent[pref.base_experiment]
            pref.save()


update_runs()
update_preferences()




