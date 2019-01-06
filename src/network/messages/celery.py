from celery import Celery
app = Celery('test_celery',
             broker='amqp://admin:mypass@rabbit:5672',
             backend='rpc://',
             include=['test_celery.tasks'])