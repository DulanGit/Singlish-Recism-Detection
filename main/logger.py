'''
Created on Apr 24, 2019

@author: dulan
'''
import logging
import os.path as osp
from params import DIR

class Logger(object):
    @staticmethod
    def get_logger(log_file_dir, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        lof_file_name = osp.join(log_file_dir, name + '.log')
        if not osp.exists(lof_file_name):
            open(lof_file_name, 'a').close()
        fh = logging.FileHandler(lof_file_name)
        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger