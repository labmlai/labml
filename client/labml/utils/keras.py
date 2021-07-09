from typing import Dict, Optional

import tensorflow as tf

from labml import tracker, logger

_MAP = {
    'size': None,
    'batch': None,
    'loss': 'loss.train',
    'accuracy': 'accuracy.train',
    'val_loss': 'loss.valid',
    'val_accuracy': 'accuracy.valid'
}


class LabMLKerasCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_batch_frequency: int = 1):
        super().__init__()
        self.save_batch_frequency = save_batch_frequency

    @staticmethod
    def _parse_logs(logs: Optional[Dict[str, any]]):
        data = {}
        if logs is None:
            return data
        for k, v in logs.items():
            if k in _MAP:
                k = _MAP[k]
            if k is None:
                continue

            data[k] = v

        return data

    def on_epoch_end(self, epoch, logs=None):
        tracker.save(self._parse_logs(logs))
        logger.log()

    def on_train_batch_end(self, batch, logs=None):
        tracker.add_global_step()
        tracker.add(self._parse_logs(logs))
        if batch % self.save_batch_frequency == 0:
            tracker.save()
