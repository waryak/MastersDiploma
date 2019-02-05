from os import environ
from celery import Celery

print("--------------- LOADING ENVIRONMENT VARIABLES ---------------")
ROLE = environ.get('ROLE')
if ROLE is None:
    raise Exception("ROLE environment variable could not be found")
else:
    print("Node role is:", ROLE)

if ROLE == "worker":
    BROKER_URL = environ.get("WORKER_BROKER_URL")
    if BROKER_URL is None:
        raise Exception("Broker URL for worker could not be found")
    else:
        print("-> Worker's broker url is:", BROKER_URL)
if ROLE == "producer":
    BROKER_URL = environ.get("PRODUCER_BROKER_URL")
    if BROKER_URL is None:
        raise Exception("Broker URL for producer could not be found")
    else:
        print("-> Worker's broker url is:", BROKER_URL)

print("-------------------- STARTING CELERY APP --------------------")
app = Celery('test_celery',
             broker=BROKER_URL,
             backend='rpc://',
             include=['src.network.messages.tasks'])






# app = Celery('test_celery',broker='amqp://admin:mypass@rabbit:5672',backend='rpc://',include=['test_celery.tasks'])