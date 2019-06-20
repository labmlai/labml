import time
from typing import List, Dict

from lab.experiment import _struct_time_to_date, _struct_time_to_time, _GLOBAL_STEP


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
                 start_step: int = 0,
                 progress=None,
                 progress_limit: int = 16):
        self.commit = commit
        self.is_dirty = is_dirty
        self.python_file = python_file
        self.trial_date = trial_date
        self.trial_time = trial_time
        self.comment = comment
        self.commit_message = commit_message
        self.start_step = start_step
        if progress is None:
            self.progress = []
        else:
            self.progress = progress

        assert progress_limit % 2 == 0
        assert progress_limit >= 8
        self.progress_limit = progress_limit

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
            start_step=self.start_step,
            progress=self.progress
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

        # Stop if no progress is available
        if len(self.progress) == 0:
            return res

        res.append('')

        # Print progress table
        lens = {}
        for k, v in self.progress[0].items():
            lens[k] = max(len(k), len(v))

        line = []
        for k, v in self.progress[0].items():
            line.append(' ' * (lens[k] - len(k)) + k)
        line = '| ' + ' | '.join(line) + ' |'
        res.append('-' * len(line))
        res.append(line)
        res.append('-' * len(line))

        for p in self.progress:
            line = [' ' * (lens[k] - len(str(v))) + str(v) for k, v in p.items()]
            line = '| ' + ' | '.join(line) + ' |'
            res.append(line)

        res.append('-' * len(line))

        return res

    def __str__(self):
        return f"Trial(comment=\"{self.comment}\"," \
            f" commit=\"{self.commit_message}\"," \
            f" date={self.trial_date}, time={self.trial_time}"

    def __repr__(self):
        return self.__str__()

    def set_progress(self, progress: Dict[str, str]):
        """
        ## Add or update progress
        """

        assert _GLOBAL_STEP in progress

        if len(self.progress) < 2:
            self.progress.append(progress)
        else:
            g0 = int(self.progress[0][_GLOBAL_STEP].replace(',', ''))
            g1 = int(self.progress[1][_GLOBAL_STEP].replace(',', ''))
            gp = int(self.progress[-2][_GLOBAL_STEP].replace(',', ''))
            gf = int(self.progress[-1][_GLOBAL_STEP].replace(',', ''))

            if g1 - g0 <= gf - gp:
                # add
                if len(self.progress) == self.progress_limit:
                    shrunk = []
                    for i in range(self.progress_limit // 2):
                        shrunk.append(self.progress[2 * i + 1])
                    self.progress = shrunk
                self.progress.append(progress)
            else:
                self.progress[-1] = progress