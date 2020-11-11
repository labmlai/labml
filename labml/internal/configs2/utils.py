import itertools
from typing import Tuple

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
