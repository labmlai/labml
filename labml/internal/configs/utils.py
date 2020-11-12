import warnings
from typing import Tuple, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from labml.internal.configs.config_item import ConfigItem

TRUNCATED_TOKEN = '[[...]]'
MIN_STR_LIMIT = 5


class Value:
    @staticmethod
    def is_primitive(value):
        if value is None:
            return True

        if (isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, bool)
        ):
            return True

        return False

    @staticmethod
    def __to_yaml_truncated_list(value, limit: int):
        res = []
        total_size = 0
        for v in value:
            r, size = Value.__to_yaml_truncated(v, limit - total_size)
            res.append(r)
            if limit - total_size < size:
                res.append(TRUNCATED_TOKEN)
                return res, limit
            total_size += size
        return res, total_size

    @staticmethod
    def __to_yaml_truncated_str(value, limit: int):
        limit = max(limit, MIN_STR_LIMIT)
        if len(value) > limit:
            res = value[:limit] + ' ' + TRUNCATED_TOKEN
            return res, limit
        else:
            return value, max(len(value), 1)

    @staticmethod
    def __to_yaml_truncated(value, limit: int) -> Tuple[any, int]:
        if isinstance(value, str):
            return Value.__to_yaml_truncated_str(value, limit)
        elif Value.is_primitive(value):
            return value, 1
        elif isinstance(value, list):
            return Value.__to_yaml_truncated_list(value, limit)
        elif isinstance(value, dict):
            res = {}
            total_size = 0
            for k, v in value.items():
                r, size = Value.__to_yaml_truncated(v, limit - total_size)
                res[k] = r
                if limit - total_size < size:
                    res[TRUNCATED_TOKEN] = TRUNCATED_TOKEN
                    return res, limit
                total_size += size
            return res, total_size
        elif isinstance(value, tuple):
            return tuple(Value.__to_yaml_truncated_list(value, limit))
        else:
            return Value.__to_yaml_truncated_str(Value.to_str(value), limit)

    @staticmethod
    def to_yaml_truncated(value, limit: int = 5000):
        return Value.__to_yaml_truncated(value, limit)[0]

    @staticmethod
    def __to_yaml(value):
        if Value.is_primitive(value):
            return value
        elif isinstance(value, list):
            return [Value.to_yaml_truncated(v) for v in value]
        elif isinstance(value, dict):
            return {k: Value.to_yaml_truncated(v) for k, v in value.items()}
        elif isinstance(value, tuple):
            return tuple(Value.to_yaml_truncated(v) for v in value)
        else:
            return Value.to_str(value)

    @staticmethod
    def to_str(value):
        str_value = str(value)
        if str_value == Value.default_repr(value):
            if value.__class__.__module__ == '__main__':
                return value.__class__.__name__
            else:
                return f"{value.__class__.__module__}.{value.__class__.__name__}"
        else:
            return str_value

    @staticmethod
    def default_repr(value):
        return '<%s.%s object at %s>' % (
            value.__class__.__module__,
            value.__class__.__name__,
            hex(id(value))
        )


def get_config_names(decorator_name: str, function_name: str,
                     config_names: Union[str, 'ConfigItem', List['ConfigItem'], List[str]], *,
                     is_check_string_names: bool = True):
    from labml.internal.configs.config_item import ConfigItem
    if config_names is None:
        warnings.warn(f"Decorate your function with @{decorator_name}(Config.{function_name})", FutureWarning, 4)
        return function_name
    elif isinstance(config_names, str):
        if is_check_string_names:
            warnings.warn(f"Use Config.{config_names} instead of '{config_names}'.", FutureWarning, 4)
        return config_names
    elif isinstance(config_names, ConfigItem):
        return config_names.key
    else:
        assert isinstance(config_names, list) and len(config_names) > 0
        keys = []
        for c in config_names:
            if isinstance(c, str):
                warnings.warn(f"Use Config.{c} instead of '{c}'", FutureWarning, 4)
                keys.append(c)
            elif isinstance(c, ConfigItem):
                keys.append(c.key)
            else:
                raise ValueError('Cannot determine config item', c)
        return keys
