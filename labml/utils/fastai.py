from fastai.callback.core import Callback, to_detach

from labml import tracker


class LabMLFastAICallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_fit(self):
        pass

    def after_batch(self):
        tracker.add_global_step()
        tracker.save({'loss.train': to_detach(self.loss.clone())})

    def after_epoch(self):
        pass

    def after_fit(self):
        pass
