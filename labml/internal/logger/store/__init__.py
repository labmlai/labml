from pathlib import PurePath
from typing import Dict, List, Any

from .artifacts import Artifact
from .indicators import Indicator, Scalar
from .namespace import Namespace
from ..writers import Writer
from ... import util


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

    def save_indicators(self, file: PurePath):
        self.__indicators_file = file

        data = {k: ind.to_dict() for k, ind in self.indicators.items() }
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
        self.__assert_name(indicator.name, indicator)
        if indicator.name.startswith('.'):
            self.dot_indicators[indicator.name] = indicator
            return

        self.indicators[indicator.name] = indicator
        indicator.clear()

        if self.__indicators_file is not None:
            self.save_indicators(self.__indicators_file)

    def add_artifact(self, artifact: Artifact):
        self.__assert_name(artifact.name, artifact)
        if artifact.name.startswith('.'):
            self.dot_artifacts[artifact.name] = artifact
            return

        self.artifacts[artifact.name] = artifact
        artifact.clear()

        if self.__artifacts_file is not None:
            self.save_artifacts(self.__artifacts_file)

    def store(self, key: str, value: any):
        suffix = key
        if key.startswith('.'):
            key = '.'.join([ns.name for ns in self.namespaces] + [key[1:]])

        if key in self.artifacts:
            self.artifacts[key].collect_value(None, value)
        elif key in self.indicators:
            self.indicators[key].collect_value(value)
        elif suffix in self.dot_indicators:
            self.add_indicator(self.dot_indicators[suffix].copy(key))
            self.indicators[key].collect_value(value)
        elif suffix in self.dot_artifacts:
            self.add_artifact(self.dot_artifacts[suffix].copy(key))
            self.artifacts[key].collect_value(None, value)
        else:
            self.add_indicator(Scalar(key, True))
            self.indicators[key].collect_value(value)

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
