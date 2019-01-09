import time
from .tasks import longtime_add


if __name__ == '__main__':

    print("----------------- STARTING SUBMITTING TASKS -----------------")
    for count in range(30):
        print("Count is ", count)
        result = longtime_add.delay(count)
        print('Task finished?', result.ready())
        print('Task result:', result.result)
        time.sleep(5)
        print('Task finished"', result.ready())
        print('Task result:', result.result)
