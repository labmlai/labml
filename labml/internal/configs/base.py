import types
import warnings
from collections import OrderedDict
from typing import Dict, List, Callable, Union, Tuple, Optional, Type, Set

from .eval_function import EvalFunction
from .config_function import ConfigFunction
from .config_item import ConfigItem
from .setup_function import SetupFunction
from ... import monit
from ...utils.errors import ConfigsError
from .utils import Value


def _is_class_method(func: Callable):
    if not isinstance(func, types.FunctionType):
        return False
    # if not callable(func):
    #     return False

    import inspect

    spec: inspect.Signature = inspect.signature(func)
    params: List[inspect.Parameter] = list(spec.parameters.values())
    if len(params) == 0:
        raise RuntimeError('Can only have methods in a config class', func)
    p = params[0]
    if p.kind != p.POSITIONAL_OR_KEYWORD:
        raise RuntimeError('Can only have methods in a config class', func)
    if p.name != 'self':
        raise RuntimeError('Can only have methods in a config class', func)

    return True


RESERVED = {'calc', 'list', 'set_hyperparams', 'set_meta', 'aggregate', 'calc_wrap', 'setup', 'to_json'}
_STANDARD_TYPES = {int, str, bool, float, Dict, List}


def _is_valid(key):
    if key.startswith('_'):
        return False

    if key in RESERVED:
        return False

    return True


def _get_base_classes(class_: Type['Configs']) -> List[Type['Configs']]:
    classes = [class_]
    level = [class_]
    next_level = []

    while len(level) > 0:
        for c in level:
            for b in c.__bases__:
                if b == object:
                    continue
                next_level.append(b)
        classes += next_level
        level = next_level
        next_level = []

    classes.reverse()

    unique_classes = []
    hashes: Set[int] = set()
    for c in classes:
        if hash(c) not in hashes:
            unique_classes.append(c)
        hashes.add(hash(c))

    return unique_classes


class PropertyKeys:
    calculators = '_calculators'
    evaluators = '_evaluators'
    hyperparams = '_hyperparams'
    aggregates = '_aggregates'
    meta = '_meta'
    setups = '_setups'


class Configs:
    _calculators: Dict[str, List[ConfigFunction]] = {}
    _evaluators: Dict[str, List[EvalFunction]] = {}
    _hyperparams: Dict[str, bool]
    _aggregates: Dict[str, Dict[str, Tuple[ConfigItem, any]]]
    _meta: Dict[str, bool]
    _setups: Dict[str, List[SetupFunction]] = {}

    __config_items: Dict[str, ConfigItem]
    __options: Dict[str, Dict[str, Union[ConfigFunction, SetupFunction]]]
    __evals: Dict[str, Dict[str, EvalFunction]]
    __types: Dict[str, Type]
    __values: Dict[str, any]
    __explicitly_specified: Set[str]
    __hyperparams: Dict[str, bool]
    __meta: Dict[str, bool]
    __aggregates: Dict[str, Dict[str, Dict[str, str]]]
    __aggregate_parent: Dict[str, str]
    __secondary_values: Dict[str, Dict[str, any]]

    __order: Dict[str, int]
    __n_calculated: int

    def __init__(self, *, _primary: str = None):
        self._primary = _primary
        self.__values = {}
        self.__cached = {}

        classes = _get_base_classes(type(self))

        self.__types = {}
        self.__options = {}
        self.__evals = {}
        self.__config_items = {}
        self.__explicitly_specified = set()
        self.__hyperparams = {}
        self.__meta = {}
        self.__aggregates = {}
        self.__aggregate_parent = {}
        self.__secondary_values = {}

        self.__order = {}
        self.__n_calculated = 0

        self.__collect_config_items(classes)
        self.__collect_calculator(classes)
        self.__collect_setup(classes)
        self.__collect_evaluator(classes)

        for c in classes:
            if PropertyKeys.hyperparams in c.__dict__:
                for k, is_hyperparam in c.__dict__[PropertyKeys.hyperparams].items():
                    self.__hyperparams[k] = is_hyperparam

        for c in classes:
            if PropertyKeys.meta in c.__dict__:
                for k, is_meta in c.__dict__[PropertyKeys.meta].items():
                    self.__meta[k] = is_meta

        for c in classes:
            if PropertyKeys.aggregates in c.__dict__:
                for k, aggregates in c.__dict__[PropertyKeys.aggregates].items():
                    self.__aggregates[k] = aggregates

        # for k, v in configs.__dict__.items():
        #     if k.startswith('_'):
        #         continue
        #
        #     if k not in self.__types:
        #         raise RuntimeError(f"Unknown key :{k}")
        #     self.__collect_value(k, v)

        # if not is_directly_specified:
        #     self.__explicitly_specified = set()
        #
        # if values is not None:
        #     for k, v in values.items():
        #         if k in self.__types:
        #             self.__collect_value(k, v)
        #         else:
        #             parts = k.split('.')
        #             if parts[0] in self.__types:
        #                 if parts[0] not in self.__secondary_values:
        #                     self.__secondary_values[parts[0]] = {}
        #                 self.__secondary_values[parts[0]][k[len(parts[0]) + 1:]] = v
        #             else:
        #                 logger.log(f'Ignoring config: {k} = {str(v)}', Text.warning)

        self.__calculate_aggregates()
        self.__calculate_missing_values()

    def __collect_config_items(self, classes: List[Type['Configs']]):
        for c in classes:
            for k, v in c.__dict__.items():
                if PropertyKeys.evaluators in c.__dict__ and k in c.__dict__[PropertyKeys.evaluators]:
                    continue
                if not _is_valid(k):
                    continue

                if v.has_value:
                    self.__values[k] = v.value

                if k in self.__config_items:
                    self.__config_items[k].update(v)
                else:
                    self.__config_items[k] = v

                if k not in self.__types:
                    self.__types[k] = v.annotation

    def __collect_calculator(self, classes: List[Type['Configs']]):
        for c in classes:
            if PropertyKeys.calculators not in c.__dict__:
                continue
            for k, calculators in c.__dict__[PropertyKeys.calculators].items():
                if k not in self.__types:
                    raise RuntimeError(f"{k} calculator is present but the config declaration is missing")
                for v in calculators:
                    if k not in self.__options:
                        self.__options[k] = OrderedDict()
                    if v.option_name in self.__options[k]:
                        if v != self.__options[k][v.option_name]:
                            warnings.warn(f"Overriding option for {k}: {v.option_name}", Warning, stacklevel=5)

                    self.__options[k][v.option_name] = v

    def __collect_setup(self, classes: List[Type['Configs']]):
        for c in classes:
            if PropertyKeys.setups not in c.__dict__:
                continue
            for k, setups in c.__dict__[PropertyKeys.setups].items():
                if k not in self.__types:
                    raise RuntimeError(f"{k} setup is present but the config declaration is missing")
                for v in setups:
                    if k not in self.__options:
                        self.__options[k] = OrderedDict()
                    if v.option_name in self.__options[k]:
                        if v != self.__options[k][v.option_name]:
                            warnings.warn(f"Overriding option for {k}: {v.option_name}", Warning, stacklevel=5)

                    self.__options[k][v.option_name] = v

    def __collect_evaluator(self, classes: List[Type['Configs']]):
        for c in classes:
            if PropertyKeys.evaluators not in c.__dict__:
                continue
            for k, evaluators in c.__dict__[PropertyKeys.evaluators].items():
                for v in evaluators:
                    if k not in self.__evals:
                        self.__evals[k] = OrderedDict()

                    self.__evals[k]['default'] = v

    def __collect_value(self, k, v):
        if not _is_valid(k):
            return

        self.__explicitly_specified.add(k)

        self.__values[k] = v
        if k not in self.__types:
            self.__types[k] = type(v)

    def __calculate_missing_values(self):
        for k in self.__types:
            if k in self.__values:
                continue

            if k in self.__options:
                self.__values[k] = next(iter(self.__options[k].keys()))
                continue

            if type(self.__types[k]) == type:
                if self.__types[k] in _STANDARD_TYPES:
                    continue

                try:
                    config_function = ConfigFunction(self.__types[k],
                                                     config_names=self.__config_items[k],
                                                     option_name='from_type')
                except ConfigsError as e:
                    continue

                self.__options[k] = OrderedDict()
                self.__options[k]['from_type'] = config_function
                self.__values[k] = 'from_type'
                continue

    def __calculate_aggregates(self):
        queue = []
        for k in self.__aggregates:
            if k not in self.__values or self.__values[k] not in self.__aggregates[k]:
                continue

            queue.append(k)

        while queue:
            k = queue.pop()
            assert k in self.__values
            option = self.__values[k]
            assert option in self.__aggregates[k]
            pairs = self.__aggregates[k][option]

            for name, opt in pairs.items():
                if name in self.__values and self.__values[name] != '__none__':
                    continue

                self.__aggregate_parent[name] = k
                self.__values[name] = opt

                if name in self.__aggregates and opt in self.__aggregates[name]:
                    queue.append(name)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
        else:
            self.__values[key] = value

    def __getattribute__(self, item):
        if item.startswith('_'):
            return object.__getattribute__(self, item)
        if item in self.__evals:
            return object.__getattribute__(self, item)

        if item not in self.__cached:
            self.__calculate(item)
        return self.__cached[item]

    def __calculate(self, item):
        self.__order[item] = self.__n_calculated
        self.__n_calculated += 1

        if item in self.__values:
            value = self.__values[item]
        elif item in self.__options:
            value = next(iter(self.__options[item].keys()))
        else:
            # Handle getting from type
            raise RuntimeError(f'Cannot calculate config: {item}')

        if item in self.__options:
            if value in self.__options[item]:
                func = self.__options[item][value]
                with monit.section(f'Prepare {item}'):
                    value = func(self)

        self.__cached[item] = value

    def __init_subclass__(cls, **kwargs):
        configs = {}

        for k, v in cls.__annotations__.items():
            if not _is_valid(k):
                continue

            configs[k] = ConfigItem(key=k,
                                    configs_class=cls,
                                    has_annotation=True, annotation=v,
                                    has_value=k in cls.__dict__,
                                    value=cls.__dict__.get(k, None))

        evals = []

        if PropertyKeys.setups not in cls.__dict__:
            cls._setups = {}

        for k, v in cls.__dict__.items():
            if not _is_valid(k):
                continue

            if _is_class_method(v):
                evals.append((k, v))
                continue

            configs[k] = ConfigItem(key=k,
                                    configs_class=cls,
                                    has_annotation=k in cls.__annotations__,
                                    annotation=cls.__annotations__.get(k, None),
                                    has_value=True, value=v)

        for e in evals:
            cls._add_eval_function(e[1], e[0])

        for k, v in configs.items():
            setattr(cls, k, v)

    @classmethod
    def _add_config_function(cls,
                             func: Callable,
                             name: Union[ConfigItem, List[ConfigItem]],
                             option: Optional[str],
                             pass_params: Optional[List[ConfigItem]]):
        if PropertyKeys.calculators not in cls.__dict__:
            cls._calculators = {}

        calc = ConfigFunction(func,
                              config_names=name,
                              option_name=option,
                              pass_params=pass_params)
        if type(calc.config_names) == str:
            config_names = [calc.config_names]
        else:
            config_names = calc.config_names

        for n in config_names:
            if n not in cls._calculators:
                cls._calculators[n] = []
            cls._calculators[n].append(calc)

    @classmethod
    def _add_eval_function(cls,
                           func: Callable,
                           name: str):
        if PropertyKeys.evaluators not in cls.__dict__:
            cls._evaluators = {}

        calc = EvalFunction(func,
                            config_name=name)

        if name not in cls._evaluators:
            cls._evaluators[name] = []
        cls._evaluators[name].append(calc)

    @classmethod
    def _add_setup_function(cls,
                            func: Callable,
                            config_names: List[ConfigItem],
                            option_name: Optional[str]):
        if PropertyKeys.setups not in cls.__dict__:
            cls._setups = {}

        calc = SetupFunction(func,
                             option_name=option_name,
                             config_names=config_names)
        config_names = calc.config_names

        for n in config_names:
            if n not in cls._setups:
                cls._setups[n] = []
            cls._setups[n].append(calc)

    @classmethod
    def calc_wrap(cls, func: Callable,
                  name: Union[ConfigItem, List[ConfigItem]],
                  option_name: Optional[str] = None,
                  pass_params: Optional[List[ConfigItem]] = None):
        cls._add_config_function(func, name, option_name, pass_params)

        return func

    @classmethod
    def setup(cls, names: Union[List[ConfigItem], ConfigItem], option_name: Optional[str]):
        if not isinstance(names, list):
            names: List[ConfigItem] = [names]

        def wrapper(func: Callable):
            cls._add_setup_function(func, names, option_name)
            return func

        return wrapper

    @classmethod
    def calc(cls,
             name: Union[ConfigItem, List[ConfigItem]] = None,
             option_name: Optional[str] = None,
             pass_params: Optional[List[ConfigItem]] = None):

        def wrapper(func: Callable):
            cls._add_config_function(func, name, option_name, pass_params)

            return func

        return wrapper

    @classmethod
    def set_hyperparams(cls, *args: ConfigItem, is_hyperparam=True):
        if PropertyKeys.hyperparams not in cls.__dict__:
            cls._hyperparams = {}

        for h in args:
            cls._hyperparams[h.key] = is_hyperparam

    @classmethod
    def set_meta(cls, *args: ConfigItem, is_meta=True):
        if PropertyKeys.meta not in cls.__dict__:
            cls._meta = {}

        for h in args:
            cls._meta[h.key] = is_meta

    @classmethod
    def aggregate(cls, name: ConfigItem, option: str,
                  *args: Tuple[ConfigItem, any]):
        assert args

        if PropertyKeys.aggregates not in cls.__dict__:
            cls._aggregates = {}

        if name.key not in cls._aggregates:
            cls._aggregates[name.key] = {}

        pairs = {p[0].key: p[1] for p in args}
        cls._aggregates[name.key][option] = pairs

    def __get_options_list(self, key: str):
        opts = list(self.__options.get(key, {}).keys())
        if not opts:
            opts = list((self.__aggregates.get(key, {}).keys()))

        return opts

    def to_json(self):
        configs = {}
        for k, v in self.__types.items():
            configs[k] = {
                'name': k,
                'type': str(v),
                'value': Value.to_yaml_truncated(self.__values.get(k, None)),
                'order': self.__order.get(k, -1),
                'options': self.__get_options_list(k),
                'computed': Value.to_yaml_truncated(self.__cached.get(k, None)),
                'is_hyperparam': self.__hyperparams.get(k, None),
                'is_meta': self.__meta.get(k, None),
                'is_explicitly_specified': (k in self.__explicitly_specified)
            }

        for k, c in self.__cached.items():
            if not isinstance(c, Configs):
                continue
            sub_configs = c.to_json()
            for sk, v in sub_configs.items():
                configs[f"{k}.{sk}"] = v

        return configs

