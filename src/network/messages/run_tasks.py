import time
import yaml
from os import environ
from messages.tasks import longtime_add
from src.network.messages.template_planner import TemplateManager

if __name__ == '__main__':

    print("-------------- LOADING PRODUCER CONFIGURATIONS --------------")
    file_conf = environ.get('CONFIG')
    try:
        with open(file_conf, 'r') as file_conf:
            configs = yaml.load(file_conf)
    except Exception:
        raise Exception("Could not find configuration file")
    NUMBER_OF_TEPLATES = int(configs["algorithm"]["number_of_templates"])
    assert (NUMBER_OF_TEPLATES > 0) & (NUMBER_OF_TEPLATES < int(10e6)), "Something wrong, with <NUMBER_OF_TEPLATES>"

    print("--------------- CREATING THE TEMPLATE MANAGER ---------------")

    tm = TemplateManager(template_size=5,
                         max_template_distance=10,
                         min_template_distance=1)



    print("----------------- STARTING SUBMITTING TASKS -----------------")
    for count in range(30):

        # TODO: HERE WE USE CSV FILE TO WRITE DOWN ALL
        template = tm.next_planned_template()

        print("Count is ", count)
        result = longtime_add.delay(count)

        print('Task finished?', result.ready())
        print('Task result:', result.result)
        time.sleep(5)
        print('Task finished"', result.ready())
        print('Task result:', result.result)

    tp = TemplateManager(size=6,
                         max_template_distance=10,
                         min_template_distance=1)



    for count in range(130):
        print("Count is ", count)
        result = longtime_add.delay(count)
        # print('Task finished?', result.ready())
        # print('Task result:', result.result)
        # print('Task finished"', result.ready())
        # print('Task result:', result.result)
