from __future__ import absolute_import
from network.celery.celery import app
import time
import datetime as dt
from pymongo import MongoClient

@app.task
def longtime_add(count):
    print('long time task begins')
    time.sleep(1)
    print('long time task finished')
    return str(count) + ". Hello. This is your timestamp " + str(dt.datetime.today())