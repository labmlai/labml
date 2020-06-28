from typing import Dict

from labml.internal.logger.store.indicators.numeric import Queue, Histogram, Scalar


def load_indicator_from_dict(data: Dict[str, any]):
    class_name = data['class_name']
    del data['class_name']

    if class_name == 'Queue':
        return Queue(**data)
    elif class_name == 'Histogram':
        return Histogram(**data)
    elif class_name == 'Scalar':
        return Scalar(**data)
    else:
        raise ValueError(f"Unknown indicator: {class_name}")
