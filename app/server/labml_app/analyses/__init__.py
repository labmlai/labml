from typing import Dict, List, Tuple, Callable

from . import analysis
from .series import SeriesModel
from ..analyses_settings import experiment_analyses, computer_analyses

EXPERIMENT_ANALYSES = {}
for ans in experiment_analyses:
    EXPERIMENT_ANALYSES[ans.__name__] = ans


class AnalysisManager:
    @staticmethod
    def track(run_uuid: str, data: Dict[str, SeriesModel]) -> None:
        for ans in experiment_analyses:
            ans.get_or_create(run_uuid).track(data, run_uuid)

    @staticmethod
    def track_computer(session_uuid: str, data: Dict[str, SeriesModel]) -> None:
        for ans in computer_analyses:
            ans.get_or_create(session_uuid).track(data)

    @staticmethod
    def delete_run(run_uuid: str) -> None:
        for ans in experiment_analyses:
            ans.delete(run_uuid)

    @staticmethod
    def delete_session(session_uuid: str) -> None:
        for ans in computer_analyses:
            ans.delete(session_uuid)

    @staticmethod
    def get_handlers() -> List[Tuple[str, Callable, str, bool]]:
        return analysis.URLS

    @staticmethod
    def get_db_indexes() -> List[Tuple[any, str]]:
        return analysis.DB_INDEXES

    @staticmethod
    def get_db_models() -> List[Tuple[any, str]]:
        return analysis.DB_MODELS

    @staticmethod
    def get_experiment_analysis(name: str, run_uuid: str) -> any:
        if name not in EXPERIMENT_ANALYSES:
            return None

        return EXPERIMENT_ANALYSES[name].get_or_create(run_uuid)
