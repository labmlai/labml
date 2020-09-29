import types
from typing import Dict, List, Callable, Union, Tuple, Optional

from labml.internal.configs.eval_function import EvalFunction
from .config_function import ConfigFunction
from .config_item import ConfigItem
from .parser import Parser, PropertyKeys
from .setup_function import SetupFunction


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


class Configs:
    _calculators: Dict[str, List[ConfigFunction]] = {}
    _evaluators: Dict[str, List[EvalFunction]] = {}
    _setups: Dict[str, List[SetupFunction]] = {}

    def __init__(self, *, _primary: str = None):
        self._primary = _primary

    def __init_subclass__(cls, **kwargs):
        configs = {}

        for k, v in cls.__annotations__.items():
            if not Parser.is_valid(k):
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
            if not Parser.is_valid(k):
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
    def _calc(cls,
              name: Union[ConfigItem, List[ConfigItem]] = None,
              option: Optional[str] = None,
              pass_params: Optional[List[ConfigItem]] = None):

        def wrapper(func: Callable):
            cls._add_config_function(func, name, option, pass_params)

            return func

        return wrapper

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
        return cls._calc(name, option_name, pass_params)

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
