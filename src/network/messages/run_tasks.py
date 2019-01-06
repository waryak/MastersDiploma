from .tasks import longtime_add
import time
if __name__ == '__main__':
    for count in range(30):
        result = longtime_add.delay(count)
        print('Task finished?',result.ready())
        print('Task result:',result.result)
        time.sleep(5)
        print('Task finished"',result.ready())
        print('Task result:',result.result)


