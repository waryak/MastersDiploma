from __future__ import absolute_import

import sys
import time
import numpy as np
import datetime as dt

from src.algo.main import main
from src.network.messages.celery import app

sys.path.append("/src")

@app.task()
def longtime_add(count):
    """
    Test function, that asks a worker to send the timestamp
    :param count: The number of task
    :return: The number of tasks and the timestamp itself
    """
    print("-------------------------------------------------------------")
    print('long time task begins', count)
    time.sleep(100)
    print('long time task finished', count)

    return str(count) + ". Hello. This is your timestamp " + str(dt.datetime.today())

@app.task
def run_wishart(template, wishart_neighbors, wishart_significance):
    """
    This tasks launches the wishart calculation on a worker.

    :param template: Numpy array  with distances between points. E.g.: [3,4,1,0,2]
    will take result in this mask: [x o o o x o o o o x o x x o o x]
    :param wishart_neighbors: Number of wishart members for clusterisation
    :param wishart_significance:
    :return:
    """
    template = [int(e) for e in template]
    wishart_neighbors = int(wishart_neighbors)
    wishart_significance = float(wishart_significance)
    clusters_completeness, significant_clusters = main(np.array(template), wishart_neighbors, wishart_significance)
    return {"Number_of_completed_clusters": clusters_completeness,
            "Number_of_significant_cluster": significant_clusters}






