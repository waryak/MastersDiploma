from __future__ import absolute_import
from messages.celery import app
import time
import datetime as dt



from src.algo.main import main
from src.algo.wishart import Wishart


@app.task()
def longtime_add(count):
    print('long time task begins')
    time.sleep(1)
    print('long time task finished')

    return str(count) + ". Hello. This is your timestamp " + str(dt.datetime.today())

@app.task
def run_parallel_wishart():
    result = main()
    return result




