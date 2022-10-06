import logging

logging.basicConfig(format = "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s", level=logging.DEBUG)

class Log:
    def __init__(self, name) -> None:
        self._logger = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(self, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._logger.warning(self, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(self, msg, *args, **kwargs)