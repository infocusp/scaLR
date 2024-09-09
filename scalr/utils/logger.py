"""This file contains an implementation of the logger in the pipeline."""

import logging


class FlowLogger(logging.Logger):
    """Class for flow logger.
    
    It logs high-level overview of pipeline execution in the terminal.
    """
    level = logging.NOTSET

    def __init__(self, name, level=None):
        if level:
            FlowLogger.level = level

        if not FlowLogger.level:
            FlowLogger.level = logging.INFO

        super().__init__(name, FlowLogger.level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s : %(message)s')

        handler = logging.StreamHandler()
        handler.setLevel(FlowLogger.level)
        handler.setFormatter(formatter)
        self.addHandler(handler)


class EventLogger(logging.Logger):
    """Class for event logger. It logs detailed file-level logs during pipeline execution.
    """
    level = logging.NOTSET
    filepath = None

    def __init__(self, name, level=None, filepath=None, stdout=False):
        """Initialize required parameters for event logger."""
        if level:
            EventLogger.level = level

        if not EventLogger.level:
            EventLogger.level = logging.INFO

        super().__init__(name, EventLogger.level)

        if filepath:
            EventLogger.filepath = filepath

        if not EventLogger.filepath:
            handler = logging.NullHandler()
        else:
            handler = logging.FileHandler(EventLogger.filepath)

        formatter = logging.Formatter('%(message)s')

        handler.setLevel(EventLogger.level)
        handler.setFormatter(formatter)
        self.addHandler(handler)

        # If user wants to print logs to stdout
        if stdout:
            handler = logging.StreamHandler()
            handler.setLevel(EventLogger.level)
            handler.setFormatter(formatter)
            self.addHandler(handler)

    def heading(self, msg, prefix, suffix, count):
        """A function to configure setting for heading."""
        self.info(f"\n{prefix*count} {msg} {suffix*count}\n")

    def heading1(self, msg):
        """A function to configure setting for heading 1."""
        self.heading(msg, "<", ">", 10)

    def heading2(self, msg):
        """A function to configure setting for heading 2."""
        self.heading(msg, "-", "-", 5)
