import os

from lab import util
from lab.experiment import experiment_run
from lab.lab import Lab


class Dir:
    def __init__(self, options):
        self.__options = {k.replace('.', '_'): v for k, v in options.items()}
        self.__list = [k for k in self.__options.keys()]

    def __dir__(self):
        return self.__list

    def __getattr__(self, k):
        return self.__options[k]


class Analyzer:
    @staticmethod
    def __get_lab():
        return Lab(os.getcwd())

    @staticmethod
    def __get_run_info(experiment_name: str, run_uuid: str):
        lab = Analyzer.__get_lab()
        experiment_path = lab.experiments / experiment_name

        run_path = experiment_path / run_uuid
        run_info_path = run_path / 'run.yaml'

        with open(str(run_info_path), 'r') as f:
            data = util.yaml_load(f.read())
            run = experiment_run.RunInfo.from_dict(experiment_path, data)

        return run

    def __init__(self, experiment_name: str, run_uuid: str, is_altair: bool = True):
        self.run_info = Analyzer.__get_run_info(experiment_name, run_uuid)
        if not is_altair:
            from .matplotlib.sqlite import MatPlotLibSQLiteAnalytics
            from .matplotlib.tb import MatPlotLibTensorBoardAnalytics

            self.tb = MatPlotLibTensorBoardAnalytics(self.run_info.tensorboard_log_path)
            self.sqlite = MatPlotLibSQLiteAnalytics(self.run_info.sqlite_path)
        else:
            from .altair.tb import AltairTensorBoardAnalytics
            self.tb = AltairTensorBoardAnalytics(self.run_info.tensorboard_log_path)

        with open(str(self.run_info.indicators_path), 'r') as f:
            self.indicators = util.yaml_load(f.read())

    def get_indicators(self, *args):
        dirs = {k: {} for k in args}

        def add(class_name, key, value):
            if class_name not in dirs:
                return
            dirs[class_name][key] = value

        for k, v in self.indicators.items():
            cn = v['class_name']
            add(cn, k, k)
            if cn == 'Histogram':
                add('Scalar', k, f"{k}.mean")
            if cn == 'Queue':
                add('Scalar', k, f"{k}.mean")
                add('Histogram', k, k)
            if cn == 'IndexedScalar':
                add('Scalar', k, k)

        return [Dir(dirs[k]) for k in args]
