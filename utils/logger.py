import os
import logging
import time
from datetime import datetime


class Log(object):

    def __init__(self, logger=None, file_dir='log', log_file_name='train'):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)
        self.log_time = datetime.now().strftime("%H:%M:%S")
        self.log_name = file_dir + "/" + log_file_name + "_" + self.log_time + '.log'

        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        fh.close()
        ch.close()

    def get_log(self):
        return self.logger

logger = Log(__name__).get_log()