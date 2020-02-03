class ConfigItem:
    def __init__(self, key: str, annotation: any, value: any):
        self.key = key
        if annotation is None and value is not None:
            annotation = type(value)
        self.annotation = annotation
        self.value = value