# -*- coding: utf-8 -*-
import sys
import os
import time
import logging

import repackage
repackage.up()

from config.logger import Logger


DEBUG = True
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
if sys.platform == "linux" or sys.platform == "linux2":
    CURRENT_PATH = "/".join(CURRENT_PATH.split("/")[:-2])
elif sys.platform == "darwin":
    CURRENT_PATH = "/".join(CURRENT_PATH.split("/")[:-2])
elif sys.platform == "win32":
    CURRENT_PATH = "\\".join(CURRENT_PATH.split("\\")[:-2])

data_path = os.path.join(CURRENT_PATH, "data")
logs_path = os.path.join(CURRENT_PATH, "logs")
model_path = os.path.join(CURRENT_PATH, "models")
temp_model_path = os.path.join(data_path, "temp")
processed_data_path = os.path.join(data_path, "processed")
converged_data_path = os.path.join(data_path, "converged")
external_data_path = os.path.join(data_path, "external")
interim_data_path = os.path.join(data_path, "interim")
generate_data_path = os.path.join(data_path, "generate")
raw_data_path = os.path.join(data_path, "raw")
visualize_path = os.path.join(CURRENT_PATH, "reports")

tps = time.strftime("%Y%m%d%H%M%S", time.gmtime())
logfile = ("_".join(["model_log", tps, ".log"]))
logfile = os.path.join(logs_path, logfile)
logger = Logger(logfile, logging.DEBUG, DEBUG)
