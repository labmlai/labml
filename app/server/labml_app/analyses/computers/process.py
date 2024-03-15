from typing import Dict, Set, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index, Key
from labml_db.serializer.pickle import PickleSerializer

from labml_app.logger import logger
from labml_app.enums import COMPUTEREnums
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences

SERIES_NAMES = ['rss', 'vms', 'cpu', 'threads', 'user', 'system']
OTHER_SERIES_NAMES = ['ppid', 'iowait']
STATIC_NAMES = ['name', 'create_time', 'pid', 'ppid', 'dead', 'exe', 'cmdline']

ALMOST_ZERO = 1.0E-2


@Analysis.db_model(PickleSerializer, 'ExperimentProcess')
class ExperimentProcess(Model['ExperimentProcessModel'], SeriesCollection):
    run_uuid: str
    process_id: str
    name: str
    create_time: float
    cmdline: str
    exe: str
    pid: int
    ppid: int
    dead: bool

    @classmethod
    def defaults(cls):
        return dict(
            run_uuid='',
            process_id='',
            name='',
            create_time=0,
            cmdline='',
            exe='',
            pid=0,
            ppid=0,
            dead=False,
        )

    def load_data(self, data):
        self.process_id = data.get('process_id', '')
        self.name = data.get('name', '')
        self.create_time = data.get('create_time', 0)
        self.cmdline = data.get('cmdline', '')
        self.exe = data.get('exe', '')
        self.pid = data.get('pid', 0)
        self.ppid = data.get('ppid', 0)
        self.dead = data.get('dead', False)
        self.run_uuid = data.get('run_uuid', '')


@Analysis.db_model(PickleSerializer, 'Process')
class ProcessModel(Model['ProcessModel'], SeriesCollection):
    names: Dict[str, str]
    exes: Dict[str, str]
    cmdlines: Dict[str, str]
    create_times: Dict[str, float]
    pids: Dict[str, float]
    ppids: Dict[str, float]
    dead: Dict[str, bool]
    gpu_processes: Dict[str, Set[str]]
    zero_cpu_processes: Dict[str, Dict['str', Any]]
    experiment_process_keys: Dict[str, Key['ExperimentProcess']]  # key: process_id

    @classmethod
    def defaults(cls):
        return dict(
            names={},
            exes={},
            cmdlines={},
            create_times={},
            pids={},
            ppids={},
            dead={},
            gpu_processes={},
            zero_cpu_processes={},
            experiment_process_keys={},
        )


@Analysis.db_index(PickleSerializer, 'process_index')
class ProcessIndex(Index['Process']):
    pass


@Analysis.db_model(PickleSerializer, 'process_preferences')
class ProcessPreferencesModel(Model['ProcessPreferencesModel'], Preferences):
    pass


@Analysis.db_index(PickleSerializer, 'process_preferences_index')
class ProcessPreferencesIndex(Index['ProcessPreferences']):
    pass


class ProcessAnalysis(Analysis):
    process: ProcessModel

    def __init__(self, data):
        self.process = data
        self.process.max_buffer_length = 100

    def add_experiment_process(self, process_id: str, process_key: Key['ExperimentProcess']):
        self.process.experiment_process_keys[process_id] = process_key
        self.process.save()

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_split = ind.split('.')
            ind_type = ind_split[0]
            if ind_type == COMPUTEREnums.PROCESS:
                suffix = ind_split[-1]
                process_id = '.'.join(ind_split[:-1])

                if 'name' == suffix:
                    if process_id not in self.process.names:
                        self.process.names[process_id] = s['value'][0]
                    continue
                elif 'exe' == suffix:
                    if process_id not in self.process.exes:
                        self.process.exes[process_id] = s['value'][0]
                    continue
                elif 'cmdline' == suffix:
                    if process_id not in self.process.cmdlines:
                        self.process.cmdlines[process_id] = s['value'][0]
                    continue
                elif 'create_time' == suffix:
                    if process_id not in self.process.create_times:
                        self.process.create_times[process_id] = s['value'][0]
                    continue
                elif 'pid' == suffix:
                    if process_id not in self.process.pids:
                        self.process.pids[process_id] = s['value'][0]
                    continue
                elif 'ppids' == suffix:
                    if process_id not in self.process.ppids:
                        self.process.ppids[process_id] = s['value'][0]
                    continue
                elif 'dead' == suffix:
                    if process_id not in self.process.dead:
                        self.process.dead[process_id] = s['value'][0]
                    continue

                if 'gpu' in process_id:
                    process_id = '.'.join(ind_split[:2])
                    gpu_process = '.'.join(ind_split[2:4])

                    if process_id in self.process.gpu_processes:
                        self.process.gpu_processes[process_id].add(gpu_process)
                    else:
                        self.process.gpu_processes[process_id] = {gpu_process}

                res[ind] = s

        for process_id in self.process.pids.keys():
            # check if there's an experiment process with a process id
            if process_id not in self.process.experiment_process_keys:
                continue
            experiment_process_key = self.process.experiment_process_keys[process_id]
            experiment_process: ExperimentProcess = experiment_process_key.load()
            if experiment_process is None:
                continue
            # filter res to have only relevant series
            experiment_process_res = {k: v for k, v in res.items() if k.startswith(process_id)}
            if experiment_process is not None:
                experiment_process.track(experiment_process_res, keep_last_24h=False)

        self.process.track(res, keep_last_24h=True)
        self.clean_dead_processes()

    def clean_dead_processes(self):
        series_names = SERIES_NAMES + OTHER_SERIES_NAMES

        process_ids_to_remove = {process_id for process_id in self.process.dead if self.process.dead[process_id]}

        inds_to_remove = {}
        for process_id in process_ids_to_remove:
            # clean out experiment processes
            if process_id in self.process.experiment_process_keys:
                experiment_process_key = self.process.experiment_process_keys[process_id]
                experiment_process: ExperimentProcess = experiment_process_key.load()
                if experiment_process is not None:
                    run_uuid = experiment_process.run_uuid
                    from ...db.run import get as get_run
                    run = get_run(run_uuid)
                    if run is not None:
                        run.session_id = ''
                        run.process_id = ''
                        run.process_key = None
                        run.save()
                    experiment_process.delete()
                self.process.experiment_process_keys.pop(process_id)

            for s in series_names:
                ind = f'{process_id}.{s}'
                inds_to_remove[ind] = process_id

        inds = list(self.process.tracking.keys())

        removed = 0
        for ind in inds:
            if ind in inds_to_remove:
                ret = self.process.tracking.pop(ind, None)
                if ret:
                    removed += 1
                    self.process.dead.pop(inds_to_remove[ind], None)

        self.process.save()
        logger.info(f'processes: {removed} number of series removed, {len(self.process.tracking)} remaining')

    def get_tracking(self):
        res = {}
        zero_cpu_processes = {}
        for ind, track in self.process.tracking.items():
            ind_split = ind.split('.')
            process_id = '.'.join(ind_split[:-1])

            dead = self.process.dead.get(process_id, 0)
            if dead:
                continue

            if process_id not in res:
                res[process_id] = {'process_id': process_id,
                                   'dead': dead,
                                   'pid': self.process.pids.get(process_id, 0),
                                   'name': self.process.names.get(process_id, ''),
                                   'is_zero_cpu': False,
                                   }

            suffix = ind_split[-1]
            if suffix in ['cpu', 'rss']:
                s = Series().load(track)

                # TODO better not to cache this and avoid smoothing for zero cpu processes
                if suffix == 'cpu' and s.summary['mean'] < ALMOST_ZERO:
                    zero_cpu_processes[process_id] = res[process_id]
                    res[process_id]['is_zero_cpu'] = True

                series: Dict[str, Any] = s.detail
                res[process_id][suffix] = series

        ret = []
        for k, v in res.items():
            if v['is_zero_cpu'] or 'cpu' not in v or 'rss' not in v:
                continue

            ret.append(v)

        ret.sort(key=lambda s: s['cpu']['value'][-1], reverse=True)

        summary = []
        for v in ret[:5]:
            v['cpu']['name'] = v['name']
            summary.append(v['cpu'])

        self.process.zero_cpu_processes = zero_cpu_processes
        self.process.save()

        return ret, summary

    def get_zero_cpu_processes(self):
        ret = []
        for k, v in self.process.zero_cpu_processes.items():
            if 'cpu' not in v or 'rss' not in v:
                continue

            ret.append(v)

        ret.sort(key=lambda s: s['cpu']['value'][-1], reverse=True)

        return ret

    def get_process(self, process_id: str):
        res = {'process_id': process_id,
               'name': self.process.names[process_id],
               'create_time': self.process.create_times.get(process_id, 0),
               'cmdline': self.process.cmdlines.get(process_id, ''),
               'exe': self.process.exes.get(process_id, ''),
               'pid': self.process.pids.get(process_id, 0),
               'ppid': self.process.ppids.get(process_id, 0),
               'dead': self.process.dead.get(process_id, 0),
               }

        series_list = []
        for s_name in SERIES_NAMES:
            ind = process_id + f'.{s_name}'

            if process_id in self.process.experiment_process_keys:
                experiment_process = self.process.experiment_process_keys[process_id].load()
                track = experiment_process.tracking.get(ind, {})
            else:
                track = self.process.tracking.get(ind, {})

            if track:
                series: Dict[str, Any] = Series().load(track).detail
                series['name'] = s_name
                series_list.append(series)

        gpu_processes = self.process.gpu_processes.get(process_id, [])
        for gpu_process in gpu_processes:
            s_name = f'{gpu_process}.mem'
            ind = f'{process_id}.{s_name}'

            track = self.process.tracking.get(ind, {})
            if track:
                series: Dict[str, Any] = Series().load(track).detail
                series['name'] = s_name
                series_list.append(series)

        res['series'] = series_list

        return res

    @staticmethod
    def get_or_create(session_uuid: str):
        process_key = ProcessIndex.get(session_uuid)

        if not process_key:
            p = ProcessModel()
            p.save()
            ProcessIndex.set(session_uuid, p.key)

            pp = ProcessPreferencesModel()
            pp.save()
            ProcessPreferencesIndex.set(session_uuid, pp.key)

            return ProcessAnalysis(p)

        return ProcessAnalysis(process_key.load())

    @staticmethod
    def delete(session_uuid: str):
        process_key = ProcessIndex.get(session_uuid)
        preferences_key = ProcessPreferencesIndex.get(session_uuid)

        if process_key:
            p: ProcessModel = process_key.load()
            ProcessIndex.delete(session_uuid)
            p.delete()

        if preferences_key:
            pp: ProcessPreferencesModel = preferences_key.load()
            ProcessPreferencesIndex.delete(session_uuid)
            pp.delete()


@Analysis.route('GET', 'process/{session_uuid}')
async def get_process_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    summary_data = []
    status_code = 404

    ans = ProcessAnalysis.get_or_create(session_uuid)
    if ans:
        track_data, summary_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': [], 'summary': summary_data})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'process/zero_cpu/{session_uuid}')
async def get_zero_cpu_processes(request: Request, session_uuid: str) -> Any:
    track_data = []
    status_code = 404

    ans = ProcessAnalysis.get_or_create(session_uuid)
    if ans:
        track_data = ans.get_zero_cpu_processes()
        status_code = 200

    response = JSONResponse({'series': track_data})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'process/{session_uuid}/details/{process_id}')
async def get_process_detail(request: Request, session_uuid: str, process_id: str) -> Any:
    data = {}
    status_code = 404

    ans = ProcessAnalysis.get_or_create(session_uuid)
    if ans:
        data = ans.get_process(process_id)
        status_code = 200

    response = JSONResponse(data)
    response.status_code = status_code

    return response


@Analysis.route('GET', 'process/preferences/{session_uuid}')
async def get_process_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = ProcessPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    pp: ProcessPreferencesModel = preferences_key.load()

    return pp.get_data()


@Analysis.route('POST', 'process/preferences/{session_uuid}')
async def set_process_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = ProcessPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    pp = preferences_key.load()
    json = await request.json()
    pp.update_preferences(json)

    logger.debug(f'update process preferences: {pp.key}')

    return {'errors': pp.errors}
