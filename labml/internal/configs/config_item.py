class ConfigItem:
    def __init__(self, key: str,
                 has_annotation: bool, annotation: any,
                 has_value: bool, value: any):
        self.key = key
        if annotation is None:
            annotation = type(value)
        self.annotation = annotation
        self.value = value
        self.has_annotation = has_annotation
        self.has_value = has_value

    def update(self, k: 'ConfigItem'):
        if k.has_annotation:
            self.has_annotation = True
            self.annotation = k.annotation

        if k.has_value:
            self.has_value = True
            self.value = k.value