import numpy as np
from os import listdir
from pickle import load
from scipy.spatial import cKDTree


from numpy.core.multiarray import ndarray


def form_cluster_centers(model, data, k):
    cluster_centers: ndarray = np.zeros(shape=(len(model.clusters_completenes), k))
    for cluster_index, cluster in enumerate(model.clusters_completenes):
        cluster_data = data[model.clusters == cluster]
        cluster_center = cluster_data.mean(axis=0)
        cluster_centers[cluster_index] = cluster_center
    return cluster_centers

def make_prediction(model, data):
    """
    :param ts:
    :param model:
     :return:
    """
    cluster_centers = model.cluster_centers
    lagged_clusters = cluster_centers[:, -1]
    kdtree = cKDTree(data=lagged_clusters)
