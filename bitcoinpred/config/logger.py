# -*- coding: utf-8 -*-
"""
@author: JulienWuthrich
"""
import logging
import functools
import time


class Logger(object):
    """Module to handle the logs of the application."""

    def __init__(self, file_name, level, to_console=False, logger_name=None):
        """Init the class.

            Args:
            -----
                file_name (str): path file
                level (obj): the type of log (DEBUG, INFO, WARNING)
                to_console (bool): show on the terminal the logs
                logger_name (str): name of the logger
        """
        if logger_name:
            self.log_name = logger_name
        else:
            self.log_name = file_name
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(level)
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s : %(levelname)s : %(message)s')
        )
        self.logger.addHandler(file_handler)
        if to_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            stream_handler.setFormatter(logging.Formatter(
                '%(name)-12s: %(asctime)s : %(levelname)s : %(message)s')
            )
            self.logger.addHandler(stream_handler)

    def log(self, lvl, message):
        """Add log into the logfile.

            Args:
            -----
                lvl (obj): type of log (INFO, DEBUG, WARNING)
                message (str): the message to dump
        """
        self.logger.log(lvl, message)

    def kill(self):
        """Kill all the logger alive."""
        for log in self.logger.handlers:
            self.logger.removeHandler(log)


def logged(func=None, level=logging.DEBUG, name=None, msg=None):
    """Decorator to log the function, with the duration.

        Args:
        -----
            func (function): the function to log
            level (logging.OBJECT): INFO, DEBUG, WARNING ...
            name (str): name of the logger
            msg (str): message to log

        Return:
        -------
            log the duration of the function
    """
    if func is None:
        return functools.partial(logged, level=level, name=name, msg=msg)

    logger = name if name else Logger(
        func.__name__ + ".log", logging.INFO)
    logmsg = msg if msg else func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper to use it as a decorator."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        msg = ":".join([str(func.__name__), str(end - start)])
        logger.log(level, logmsg)
        logger.log(level, msg)

        return result

    return wrapper
