import logging


class FlowLogger(logging.Logger):

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

    level = logging.NOTSET
    filepath = None

    def __init__(self, name, level=None, filepath=None):
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

    def heading(self, str):
        msg = '\n'
        msg += ''.join(['<' for _ in range(10)])
        msg += ' ' + str + ' '
        msg += ''.join(['>' for _ in range(10)])
        msg += '\n'
        self.info(msg)
