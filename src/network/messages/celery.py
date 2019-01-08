from os import environ
from celery import Celery

print("--------------- LOADING ENVIRONMENT VARIABLES ---------------")
PRODUCER_BROKER_URL = environ.get('CONFIG')
if PRODUCER_BROKER_URL is None:
    raise Exception("Broker URL for producer could not be found")
else:
    print("-> Producer's url is:", PRODUCER_BROKER_URL)
WORKER_BROKER_URL = environ.get('CONFIG')
if WORKER_BROKER_URL is None:
    raise Exception("Broker URL for worker could not be found")
else:
    print("-> Worker's url is:", WORKER_BROKER_URL)
app = Celery('test_celery',
             broker=PRODUCER_BROKER_URL,
             backend='rpc://',
             include=['messages.tasks'])






# app = Celery('test_celery',broker='amqp://admin:mypass@rabbit:5672',backend='rpc://',include=['test_celery.tasks'])