class Value:
    @staticmethod
    def is_primitive(value):
        if value is None:
            return True

        if type(value) == str:
            return True

        if type(value) == int:
            return True

        if type(value) == bool:
            return True

        if type(value) == list and all([Value.is_primitive(v) for v in value]):
            return True

        if type(value) == dict and all([Value.is_primitive(v) for v in value.values()]):
            return True

        return False

    @staticmethod
    def to_yaml(value):
        if Value.is_primitive(value):
            return value
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
