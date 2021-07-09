# Reference: https://github.com/benoitc/gunicorn/blob/master/examples/example_config.py
# import os
import multiprocessing

# _ROOT = os.path.dirname(os.path.abspath("__file__"))
# _ETC = os.path.join(_ROOT, 'etc')

loglevel = 'info'

errorlog = '../logs/api-error.log'
accesslog = '../logs/api-access.log'

bind = '0.0.0.0:5000'
workers = 2  # multiprocessing.cpu_count() * 2 + 1
threads = 8

timeout = 3 * 60  # 3 minutes
keepalive = 24 * 60 * 60  # 1 day

capture_output = True
