import time
from typing import List, Dict


def _struct_time_to_time(t: time.struct_time):
    return f"{t.tm_hour :02}:{t.tm_min :02}:{t.tm_sec :02}"


def _struct_time_to_date(t: time.struct_time):
    return f"{t.tm_year :04}-{t.tm_mon :02}-{t.tm_mday :02}"


_GLOBAL_STEP = 'global_step'


class Trial:
    """
    # Trial 🏃‍

    Every trial in an experiment has same configs.
    It's just multiple runs.

    A new trial will replace checkpoints and TensorBoard summaries
    or previous trials, you should make a copy if needed.
    The performance log in `trials.yaml` is not replaced.

    You should run new trials after bug fixes or to see performance is
     consistent.

    If you want to try different configs, create multiple experiments.
    """

    progress: List[Dict[str, str]]

    def __init__(self, *,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 comment: str,
                 commit: str or None = None,
                 commit_message: str or None = None,
                 is_dirty: bool = True,
                 diff: str or None = None,
                 start_step: int = 0):
        self.index = -1
        self.commit = commit
        self.is_dirty = is_dirty
        self.diff = diff
        self.python_file = python_file
        self.trial_date = trial_date
        self.trial_time = trial_time
        self.comment = comment
        self.commit_message = commit_message
        self.start_step = start_step

    @classmethod
    def new_trial(cls, *,
                  python_file: str,
                  trial_time: time.struct_time,
                  comment: str):
        """
        ## Create a new trial
        """
        return cls(python_file=python_file,
                   trial_date=_struct_time_to_date(trial_time),
                   trial_time=_struct_time_to_time(trial_time),
                   comment=comment)

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        """
        ## Create a new trial from a dictionary
        """
        return cls(**data)

    def to_dict(self):
        """
        ## Convert trial to a dictionary for saving
        """
        return dict(
            python_file=self.python_file,
            trial_date=self.trial_date,
            trial_time=self.trial_time,
            comment=self.comment,
            commit=self.commit,
            commit_message=self.commit_message,
            is_dirty=self.is_dirty,
            start_step=self.start_step
        )

    def pretty_print(self) -> List[str]:
        """
        ## 🎨 Pretty print trial for the python file header
        """

        # Trial information
        commit_status = "[dirty]" if self.is_dirty else "[clean]"
        res = [
            f"{self.trial_date} {self.trial_time}",
            self.comment,
            f"[{commit_status}]: {self.commit_message}",
            f"start_step: {self.start_step}"
        ]

        return res

    def __str__(self):
        return f"Trial(comment=\"{self.comment}\"," \
            f" commit=\"{self.commit_message}\"," \
            f" date={self.trial_date}, time={self.trial_time}"

    def __repr__(self):
        return self.__str__()