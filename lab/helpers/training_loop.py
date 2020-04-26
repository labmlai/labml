from lab.internal.helpers.training_loop import TrainingLoop
from lab.configs import BaseConfigs


class TrainingLoopConfigs(BaseConfigs):
    loop_count: int = 10
    loop_step: int = 1
    is_save_models: bool = False
    log_new_line_interval: int = 1
    log_write_interval: int = 1
    save_models_interval: int = 1
    is_loop_on_interrupt: bool = True

    training_loop: TrainingLoop


@TrainingLoopConfigs.calc(TrainingLoopConfigs.training_loop)
def _loop_configs(c: TrainingLoopConfigs):
    return TrainingLoop(loop_count=c.loop_count,
                        loop_step=c.loop_step,
                        is_save_models=c.is_save_models,
                        log_new_line_interval=c.log_new_line_interval,
                        log_write_interval=c.log_write_interval,
                        save_models_interval=c.save_models_interval,
                        is_loop_on_interrupt=c.is_loop_on_interrupt)


