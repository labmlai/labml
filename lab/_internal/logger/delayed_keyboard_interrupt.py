import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lab._internal.logger.internal import LoggerInternal

from lab.logger import Text


class DelayedKeyboardInterrupt:
    """
    ### Capture `KeyboardInterrupt` and fire it later
    """

    def __init__(self, logger: 'LoggerInternal'):
        self.signal_received = None
        self.logger = logger

    def __enter__(self):
        self.signal_received = None
        # Start capturing
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        # Pass second interrupt without delaying
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)
            return

        # Store the interrupt signal for later
        self.signal_received = (sig, frame)
        self.logger.log([('\nSIGINT received. Delaying KeyboardInterrupt.',
                          Text.danger)])

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset handler
        signal.signal(signal.SIGINT, self.old_handler)

        # Pass on any captured interrupt signals
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)
