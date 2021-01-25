import types
import warnings
from collections import OrderedDict
from typing import Dict, List, Callable, Union, Tuple, Optional, Type, Set, Iterable, Any

from .config_function import ConfigFunction
from .config_item import ConfigItem
from .eval_function import EvalFunction
from .utils import Value
from ... import monit


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


RESERVED_CLASS = {'calc', 'list', 'set_hyperparams', 'set_meta', 'aggregate', 'calc_wrap'}
RESERVED_INSTANCE = {'set_default', '_to_json', '_reset_explicitly_specified', '_set_update_callback',
                     '_set_values', '_get_type'}

_STANDARD_TYPES = {int, str, bool, float, Dict, List}


def _is_valid(key):
    if key.startswith('_'):
        return False

    if key in RESERVED_CLASS or key in RESERVED_INSTANCE:
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


class Configs:
    _calculators: Dict[str, List[ConfigFunction]] = {}
    _evaluators: Dict[str, List[EvalFunction]] = {}
    _hyperparams: Dict[str, bool]
    _aggregates: Dict[str, Dict[str, Dict[ConfigItem, any]]]
    _meta: Dict[str, bool]

    __config_items: Dict[str, ConfigItem]
    __options: Dict[str, Dict[str, ConfigFunction]]
    __evals: Dict[str, Dict[str, EvalFunction]]
    __types: Dict[str, Type]
    __explicitly_specified: Set[str]
    __hyperparams: Dict[str, bool]
    __meta: Dict[str, bool]
    __aggregates: Dict[str, Dict[str, Dict[str, any]]]
    __aggregates_options: Dict[str, Set[str]]
    __secondary_values: Dict[str, Dict[str, any]]

    __defaults: Dict[str, any]
    __values: Dict[str, any]
    __values_override: Dict[str, any]
    __default_overrides: Dict[str, Tuple[any, Callable]]
    __cached: Dict[str, any]
    __cached_configs: Dict[str, 'Configs']

    __order: Dict[str, int]
    __n_calculated: int

    __update_callback: Optional[Callable]

    def __init__(self, *, _primary: str = None):
        self._primary = _primary
        self.__defaults = {}
        self.__values = {}
        self.__values_override = {}
        self.__default_overrides = {}
        self.__cached = {}
        self.__cached_configs = {}

        classes = _get_base_classes(type(self))

        self.__types = {}
        self.__options = {}
        self.__evals = {}
        self.__config_items = {}
        self.__explicitly_specified = set()
        self.__hyperparams = {}
        self.__meta = {}
        self.__aggregates = {}
        self.__aggregates_options = {}
        self.__secondary_values = {}

        self.__order = {}
        self.__n_calculated = 0

        self.__update_callback = None

        self.__collect_config_items(classes)
        self.__collect_calculator(classes)
        self.__collect_evaluator(classes)
        self.__collect_aggregates(classes)

        for c in classes:
            if PropertyKeys.hyperparams in c.__dict__:
                for k, is_hyperparam in c.__dict__[PropertyKeys.hyperparams].items():
                    self.__hyperparams[k] = is_hyperparam

        for c in classes:
            if PropertyKeys.meta in c.__dict__:
                for k, is_meta in c.__dict__[PropertyKeys.meta].items():
                    self.__meta[k] = is_meta

    def __collect_config_items(self, classes: List[Type['Configs']]):
        for c in classes:
            for k, v in c.__dict__.items():
                if PropertyKeys.evaluators in c.__dict__ and k in c.__dict__[PropertyKeys.evaluators]:
                    continue
                if not _is_valid(k):
                    continue

                if v.has_value:
                    self.__defaults[k] = v.value

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

    def __collect_evaluator(self, classes: List[Type['Configs']]):
        for c in classes:
            if PropertyKeys.evaluators not in c.__dict__:
                continue
            for k, evaluators in c.__dict__[PropertyKeys.evaluators].items():
                for v in evaluators:
                    if k not in self.__evals:
                        self.__evals[k] = OrderedDict()

                    self.__evals[k]['default'] = v

    def __collect_aggregates(self, classes: List[Type['Configs']]):
        for c in classes:
            if PropertyKeys.aggregates not in c.__dict__:
                continue
            for key, aggregates in c.__dict__[PropertyKeys.aggregates].items():
                for option, pairs in aggregates.items():
                    if key not in self.__aggregates_options:
                        self.__aggregates_options[key] = set()
                    self.__aggregates_options[key].add(option)
                    for name, value in pairs.items():
                        if name not in self.__aggregates:
                            self.__aggregates[name] = {}
                        if key not in self.__aggregates[name]:
                            self.__aggregates[name][key] = {}
                        self.__aggregates[name][key][option] = value

    def __dir__(self) -> Iterable[str]:
        return [k for k in self.__types]

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
        elif key in self.__types:
            if key in self.__cached:
                raise ValueError(f"Cannot set {self.__class__.__name__}:{key} after it was accessed")
            self.__values[key] = value
        else:
            raise AttributeError(f"{self.__class__.__name__} has no annotation for attribute '{key}'")

    def __getattribute__(self, item):
        if item.startswith('_'):
            return object.__getattribute__(self, item)
        if item in RESERVED_INSTANCE:
            return object.__getattribute__(self, item)
        if item in self.__evals:
            return object.__getattribute__(self, item)

        if item not in self.__cached:
            self.__calculate(item)
            if self.__update_callback is not None:
                self.__update_callback()
        return self.__cached[item]

    def __get_value_aggregate(self, item):
        if item not in self.__aggregates:
            return None, False

        for k, options in self.__aggregates[item].items():
            v, has = self.__get_value(k)
            if has and v in options:
                if k not in self.__cached:
                    self.__calculate(k)
                return options[v], True

        return None, False

    def __get_value_direct(self, item: str) -> Tuple[Any, bool]:
        if item in self.__values_override:
            return self.__values_override[item], True

        if item in self.__values:
            return self.__values[item], True

        if item in self.__default_overrides:
            v, f = self.__default_overrides[item]
            if f is None:
                return v, True
            else:
                return f(), True

        return None, False

    def __get_value(self, item):
        if item in self.__cached:
            return self.__cached[item], True

        value, has = self.__get_value_direct(item)

        if has and value != '__aggregate__':
            return value, has

        value, has = self.__get_value_aggregate(item)
        if has:
            return value, has

        if item in self.__defaults:
            return self.__defaults[item], True

        if item in self.__options:
            return next(iter(self.__options[item].keys())), True

        return None, False

    def __calculate(self, item):
        self.__order[item] = self.__n_calculated
        self.__n_calculated += 1

        value, has = self.__get_value(item)

        if has:
            pass
        elif item not in self.__types:
            raise AttributeError(f"{self.__class__.__name__} has no attribute `{item}`")
        elif type(self.__types[item]) == type and self.__types[item] not in _STANDARD_TYPES:
            config_function = ConfigFunction(self.__types[item],
                                             config_names=self.__config_items[item],
                                             option_name='from_type')

            self.__options[item] = OrderedDict()
            self.__options[item]['from_type'] = config_function
            self.__defaults[item] = 'from_type'
            value = 'from_type'
        else:
            raise AttributeError(f"{self.__class__.__name__} cannot calculate config `{item}`")

        if item in self.__options and value in self.__options[item]:
            func = self.__options[item][value]
            with monit.section(f'Prepare {item}'):
                value = func(self)

        if isinstance(value, Configs):
            self.__cached_configs[item] = value
            value._reset_explicitly_specified()
            if item in self.__secondary_values:
                s_values = self.__secondary_values[item]
                del self.__secondary_values[item]
                value._set_values(s_values)

            primary = value.__dict__.get('_primary', None)
            if primary is not None:
                value = getattr(value, primary)

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
    def calc_wrap(cls, func: Callable,
                  name: Union[ConfigItem, List[ConfigItem]],
                  option_name: Optional[str] = None,
                  pass_params: Optional[List[ConfigItem]] = None):
        cls._add_config_function(func, name, option_name, pass_params)

        return func

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

    def _set_values(self, values: Dict[str, any]):
        sub_modules = set()
        for k, v in values.items():
            if not _is_valid(k):
                raise KeyError(f"Invalid attribute name '{k}'")
            if k in self.__types:
                if k in self.__cached:
                    raise ValueError(f"Cannot set {self.__class__.__name__}:{k} after it was accessed")

                self.__explicitly_specified.add(k)

                self.__values_override[k] = v
            else:
                tk, *_ = k.split('.')
                sub_modules.add(tk)
                if tk not in self.__types:
                    raise KeyError(f"Invalid attribute name '{tk}' (taken from '{k}')")

                if tk not in self.__secondary_values:
                    self.__secondary_values[tk] = {}
                self.__secondary_values[tk][k[len(tk) + 1:]] = v

        for k in sub_modules:
            if k in self.__cached_configs:
                s_values = self.__secondary_values[k]
                del self.__secondary_values[k]
                self.__cached_configs[k]._set_values(s_values)

    def __get_options_list(self, key: str):
        opts = list(self.__options.get(key, {}).keys())
        if not opts:
            opts = list((self.__aggregates_options.get(key, set())))

        return opts

    def _to_json(self):
        configs = {}
        for k, v in self.__types.items():
            configs[k] = {
                'name': k,
                'type': str(v),
                'value': Value.to_yaml_truncated(self.__values_override.get(k, self.__values.get(k, None))),
                'order': self.__order.get(k, -1),
                'options': self.__get_options_list(k),
                'computed': Value.to_yaml_truncated(self.__cached.get(k, None)),
                'is_hyperparam': self.__hyperparams.get(k, None),
                'is_meta': self.__meta.get(k, None),
                'is_explicitly_specified': (k in self.__explicitly_specified)
            }

        for k, c in self.__cached_configs.items():
            sub_configs = c._to_json()
            for sk, v in sub_configs.items():
                configs[f"{k}.{sk}"] = v

        return configs

    def _reset_explicitly_specified(self):
        self.__explicitly_specified = set()
        for k, v in self.__cached_configs.items():
            v._reset_explicitly_specified()

    def _set_update_callback(self, update_callback: Callable):
        self.__update_callback = update_callback

    def _get_type(self, item: str) -> Type:
        return self.__types[item]

    def set_default(self, item: Any, value: Any = None, func: Optional[Callable] = None):
        if isinstance(item, ConfigItem):
            item = item.key
        else:
            warnings.warn(f"Use Config.{item} when setting defaults", FutureWarning, 4)

        self.__default_overrides[item] = (value, func)
