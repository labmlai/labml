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
    def to_yaml(value):
        if Value.is_primitive(value):
            return value
        elif isinstance(value, list):
            return [Value.to_yaml(v) for v in value]
        elif isinstance(value, dict):
            return {k: Value.to_yaml(v) for k, v in value.items()}
        elif isinstance(value, tuple):
            return tuple(Value.to_yaml(v) for v in value)
        else:
            return Value.to_str(value)

    @staticmethod
    def to_str(value):
        if str(value) == Value.default_repr(value):
            if value.__class__.__module__ == '__main__':
                return value.__class__.__name__
            else:
                return f"{value.__class__.__module__}.{value.__class__.__name__}"
        else:
            return str(value)

    @staticmethod
    def default_repr(value):
        return '<%s.%s object at %s>' % (
            value.__class__.__module__,
            value.__class__.__name__,
            hex(id(value))
        )
