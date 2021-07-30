from fastai.callback.core import Callback, to_detach, patch
from fastai.learner import Learner

from labml import tracker, experiment


class LabMLFastAICallback(Callback):
    """
    FastAI callback integration.
    Pass an instance of this class to FastAI learner as argument ``cbs``.
    FastAI will call relavent mehtods of this class to log metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_fit(self):
        pass

    def after_batch(self):
        tracker.add_global_step()
        if self.training:
            metrics = {'loss.train': self.learn.loss}
        else:
            metrics = {'loss.valid': self.learn.loss}
        try:
            for m in self.learn.metrics:
                if m.value is not None:
                    metrics[m.name] = m.value
        except:
            pass

        tracker.save(metrics)

    def after_epoch(self):
        metrics = {}
        try:
            for m in self.learn.metrics:
                if m.value is not None:
                    metrics[m.name] = m.value
        except:
            pass

        tracker.save(metrics)
        tracker.new_line()

    def after_fit(self):
        pass


@patch
def labml_configs(self: Learner):
    configs = {}
    try:
        configs['n_epoch'] = self.learn.n_epoch
        configs['model_class'] = str(type(self.learn.model))
    except:
        pass

    return configs
