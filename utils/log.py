import logging
import os
import sys

logging.basicConfig(format = "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s", level=logging.DEBUG)

class Log:
    def __init__(self, name, log_dir='log/') -> None:
        self._logger = logging.getLogger(name)

        self.make_if_not_exists(log_dir)
        self.file_handler = logging.FileHandler(
            filename=os.path.join(log_dir, 'experiment.log')
        )
        self.stdout_handler = logging.StreamHandler(sys.stdout)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(self, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._logger.warning(self, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(self, msg, *args, **kwargs)

    def make_if_not_exists(self, new_dir): 
        if not os.path.exists(new_dir): 
            os.system('mkdir -p {}'.format(new_dir))