import time
import yaml
from os import environ
from src.network.messages.tasks import longtime_add, run_wishart
from src.network.messages.template_planner import TemplateManager

if __name__ == '__main__':

    print("-------------- LOADING PRODUCER CONFIGURATIONS --------------")
    file_conf = environ.get('CONFIG')
    try:
        with open(file_conf, 'r') as file_conf:
            configs = yaml.load(file_conf)
    except Exception:
        raise Exception("Could not find configuration file")
    NUMBER_OF_TEMPLATES = int(configs["algorithm"]["number_of_templates"])
    assert (NUMBER_OF_TEMPLATES > 0) & (NUMBER_OF_TEMPLATES < int(10e6)), "Something wrong, with <NUMBER_OF_TEPLATES>"

    print("--------------- CREATING THE TEMPLATE MANAGER ---------------")

    tm = TemplateManager(template_size=4,
                         max_template_distance=10,
                         min_template_distance=1)

    print("----------------- STARTING SUBMITTING TASKS -----------------")

    for count in range(130):
        # TODO: HERE WE USE CSV FILE TO WRITE DOWN ALL TEMPLATES USED CHECK IF ARE NOT USING DUPLICATED TEMPLATE

        template = tm.next_planned_template(method="concurrent",
                                            step=3)
        wishart_neighbors = tm.next_planned_neighbors()
        wishart_significance = tm.next_planned_significance()
        # TODO: Arrows are printed with extra space - look inside src.algo.main
        print("That's the task number %i\n" % count,
              "-> Running with parameters:\n"
              "--> wishart_neighbors=%s\n" % wishart_neighbors,
              "--> wishart_significance=%s" % wishart_significance)

        # TODO: BULLSHIT WITH SERIALIZATION. NEED TO FIX ARGUMENT PASSING TO TASK
        template = list(template)
        template = [str(e) for e in template]
        result = run_wishart.delay(template, str(wishart_neighbors), str(wishart_significance))

        # result = longtime_add.delay(count)

        print('Task finished?', result.ready())
        print('Task result:', result.result)
        time.sleep(30)
        print('Task finished"', result.ready())
        print('Task result:', result.result)




    # for count in range(130):
    #     print("Count is ", count)
    #     result = longtime_add.delay(count)
        # print('Task finished?', result.ready())
        # print('Task result:', result.result)
        # print('Task finished"', result.ready())
        # print('Task result:', result.result)


    def preprocess(d):
        d["is_click"] = (d.ev_type == "CLICK").astype(int)
        d = d.rename(columns={"visitor_id": "client_data",
                              "ts_event": "date_view"})
        d = d.loc[:, ["client_data", "campaign", "date_view", "is_click"]]
        return d