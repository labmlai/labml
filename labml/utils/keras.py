import tensorflow as tf

from labml import tracker, logger


class LabMLKerasCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_batch_frequency: int = 1):
        super().__init__()
        self.save_batch_frequency = save_batch_frequency

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        tracker.save(logs)
        logger.log()

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        tracker.add_global_step()
        if 'size' in logs:
            del logs['size']
        if 'batch' in logs:
            del logs['batch']
        tracker.add(logs)
        if batch % self.save_batch_frequency == 0:
            tracker.save()
