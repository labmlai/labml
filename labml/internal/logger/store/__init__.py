from pathlib import PurePath
from typing import Dict, List, Any, Iterable

from .artifacts import Artifact
from .indicators import Indicator, Scalar
from .namespace import Namespace
from ..writers import Writer
from ... import util
from ...util import strings


class Store:
    dot_artifacts: Dict[str, Artifact]
    dot_indicators: Dict[str, Indicator]
    namespaces: List[Namespace]
    artifacts: Dict[str, Artifact]
    indicators: Dict[str, Indicator]

    def __init__(self):
        self.indicators = {}
        self.artifacts = {}
        self.dot_indicators = {}
        self.dot_artifacts = {}
        self.__indicators_file = None
        self.__artifacts_file = None
        self.namespaces = []
        self.add_indicator(Scalar('*', True))

    def save_indicators(self, file: PurePath):
        self.__indicators_file = file

        data = {k: ind.to_dict() for k, ind in self.indicators.items()}
        with open(str(file), "w") as file:
            file.write(util.yaml_dump(data))

    def save_artifacts(self, file: PurePath):
        self.__artifacts_file = file

        data = {k: art.to_dict() for k, art in self.artifacts.items()}
        with open(str(file), "w") as file:
            file.write(util.yaml_dump(data))

    def __assert_name(self, name: str, value: any):
        if name.startswith("."):
            if name in self.dot_indicators:
                assert self.dot_indicators[name].equals(value)
            if name in self.dot_artifacts:
                assert self.dot_artifacts[name].equals(value)

        assert name not in self.indicators, f"{name} already used"
        assert name not in self.artifacts, f"{name} already used"

    def namespace_enter(self, ns: Namespace):
        self.namespaces.append(ns)

    def namespace_exit(self, ns: Namespace):
        if len(self.namespaces) == 0:
            raise RuntimeError("Impossible")

        if ns is not self.namespaces[-1]:
            raise RuntimeError("Impossible")

        self.namespaces.pop(-1)

    def add_indicator(self, indicator: Indicator):
        self.dot_indicators[indicator.name] = indicator

        if self.__indicators_file is not None:
            self.save_indicators(self.__indicators_file)

    def add_artifact(self, artifact: Artifact):
        self.dot_artifacts[artifact.name] = artifact

        if self.__artifacts_file is not None:
            self.save_artifacts(self.__artifacts_file)

    def store(self, key: str, value: any):
        if key.startswith('.'):
            key = '.'.join([ns.name for ns in self.namespaces] + [key[1:]])

        if key in self.artifacts:
            self.artifacts[key].collect_value(None, value)
        elif key in self.indicators:
            self.indicators[key].collect_value(value)
        else:
            art_key, art_score = strings.match(key, self.dot_artifacts.keys())
            ind_key, ind_score = strings.match(key, self.dot_indicators.keys())

            if art_score > ind_score:
                self.artifacts[key] = self.dot_artifacts[art_key].copy(key)
                if self.__artifacts_file is not None:
                    self.save_artifacts(self.__artifacts_file)
            else:
                self.indicators[key] = self.dot_indicators[ind_key].copy(key)
                if self.__indicators_file is not None:
                    self.save_indicators(self.__indicators_file)

            self.store(key, value)

    def clear(self):
        for k, v in self.indicators.items():
            v.clear()
        for k, v in self.artifacts.items():
            v.clear()

    def write(self, writer: Writer, global_step):
        return writer.write(global_step=global_step,
                            indicators=self.indicators,
                            artifacts=self.artifacts)

    def create_namespace(self, name: str):
        return Namespace(store=self,
                         name=name)
