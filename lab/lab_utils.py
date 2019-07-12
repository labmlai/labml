from typing import List

from lab import colors, util, Lab
from lab.experiment import ExperimentInfo, Trial
from lab.logger import Logger


def list_experiments(lab: Lab, logger: Logger):
    experiments = lab.get_experiments()
    names = [e.name for e in experiments]
    names.sort()
    logger.info(names)


def list_trials(trials: List[Trial], logger: Logger):
    for trial in trials:
        commit_message = trial.commit_message.replace("\n", "\\n")
        logger.log_color([
            (trial.trial_date, colors.BrightColor.cyan),
            (" ", None),
            (trial.trial_time, colors.BrightColor.cyan),
            (" : ", None),
            (f"\"{trial.comment}\"", colors.BrightColor.orange),
            (" commit=", None),
            (f"\"{commit_message}\"", colors.BrightColor.purple),
        ])


def get_tensorboard_cmd(lab: Lab, experiments: List[str]):
    log_dirs = []
    for exp_name in experiments:
        exp = ExperimentInfo(lab, exp_name)
        if not exp.exists():
            raise Exception(f"Experiment {exp_name} does not exist")

        log_dirs.append(f"{exp_name}:{exp.summary_path}")

    return f"tensorboard --logdir={','.join(log_dirs)}"


def get_trials(lab: Lab, exp_name: str):
    exp = ExperimentInfo(lab, exp_name)
    trials = []
    with open(exp.trials_log_file, "r") as file:
        content = file.read()
        trials_dict = util.yaml_load(content)
        for d in trials_dict:
            trial = Trial.from_dict(d)
            trials.append(trial)

    return trials


def get_last_trials(lab: Lab, experiments: List[str]) -> List[Trial]:
    exp_trials = []
    for exp_name in experiments:
        trials = get_trials(lab, exp_name)
        exp_trials.append(trials[-1])

    return exp_trials
