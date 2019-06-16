from lab import logger


class Progress:
    """
    ### Manually monitor percentage progress.
    """

    def __init__(self, total: float, *,
                 logger: 'logger.Logger'):
        self.total = total
        self.logger = logger
        self.logger.log(f" {0.0 :4.0f}%", new_line=False)

    def update(self, value):
        """
        Update progress
        """
        percentage = min(1, max(0, value / self.total)) * 100
        self.logger.pop_current_line()
        self.logger.log(f" {percentage :4.0f}%",
                        new_line=False)

    def clear(self):
        """
        Clear line
        """
        self.logger.pop_current_line()
