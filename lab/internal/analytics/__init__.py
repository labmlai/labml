import lab
from lab.internal import util
from lab.internal.experiment import experiment_run


class Dir:
    def __init__(self, options):
        self.__options = {k.replace('.', '_'): v for k, v in options.items()}
        self.__list = [k for k in self.__options.keys()]

    def __dir__(self):
        return self.__list

    def __getattr__(self, k):
        return self.__options[k]


class Analyzer:
    r"""
    This is used to analyze runs.
    It can be used to fetch information about runs
    such as configs and logged values.

    Arguments:
        experiment_name (str): name of the experiment
        run_uuid (str): UUID of the run. You can
            get this from `dashboard <https://github.com/lab-ml/lab_dashboard>`_
        is_altair (bool): whether to use `altair <https://altair-viz.github.io>`_
            for visualizations.
            If ``False`` *Matplotlib* will be used instead.

    Attributes:
        tb (AltairTensorBoardAnalytics or MatPlotLibTensorBoardAnalytics): analytics
            based on Tensorboard logs

    Example:
        >>> a = Analyzer('mnist_loop', '1d3f855874d811eabb9359457a24edc8')
    """

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

    @staticmethod
    def __get_run_info(experiment_name: str, run_uuid: str):
        experiment_path = lab.get_experiments_path() / experiment_name

        run_path = experiment_path / run_uuid
        run_info_path = run_path / 'run.yaml'

        with open(str(run_info_path), 'r') as f:
            data = util.yaml_load(f.read())
            run = experiment_run.RunInfo.from_dict(experiment_path, data)

        return run


    def get_indicators(self, *args: str):
        r"""
        Get different types of indicators

        Arguments:
            args (str): list of types of indicators to retrieve.
                The valid types are ``Histogram``, ``Scalar``,
                ``Queue``, and ``IndexedScalar``.

        Returns:
            returns a list of objects for each type of indicators.

        Example:
            >>> Histograms, Scalars = a.get_indicators('Histogram', 'Scalar')
            >>> dir(Histograms)
        """

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
