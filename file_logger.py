
from logger import logger

class file_logger(logger):

    def __init__(self, log_level, file='file_log.txt'):
        super().__init__(log_level)
        self.file = file

    def log(self, log_level, message):
        if log_level <= self.__log_level__:
            with open(self.file, 'a') as f:
                f.write(message)
                f.write('\n')
