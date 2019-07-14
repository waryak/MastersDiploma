import argparse
import numpy as np

from os import listdir
from pickle import load
from pathlib import PurePosixPath
from os.path import isfile, isdir
from scipy.spatial import cKDTree

from src.datamart.utils import parse_model_name
from src.datamart.data_preprocesser import DataPreprocessor
from src.algo.wishart import construct_distance_matrix, sort_z_vectors_by_neighbors, density_near_zvector

parser = argparse.ArgumentParser(description='Paths to 1)initial time-series')

parser.add_argument('--ts_path',
                    dest="path_to_ts",
                    required=True,
                    default=None,
                    type=str,
                    help='Path to initial file with time-series')

parser.add_argument('--models_path',
                    dest="path_to_models",
                    required=True,
                    default=None,
                    type=str,
                    help='Path to directory with models')

parser.add_argument('--new_models_path',
                    dest="new_path_to_models",
                    required=True,
                    default=None,
                    type=str,
                    help='New Path to directory with new models')

args = parser.parse_args()

# ---------------------------
# Work with time-series file
# ---------------------------

path_to_ts = args.path_to_ts
if isfile(path_to_ts):
    print("File with time-series seems to be OK")
else:
    raise Exception("GO FUCKING CHECK PATH TO TIME SERIES")
with open(path_to_ts, "rb") as f:
    initial_ts = np.load(f)

# ---------------------------
# Work with models directory
# ---------------------------

path_to_models = args.path_to_models
if isdir(path_to_models):
    print("Directory with models seems to be OK")
else:
    raise Exception("GO FUCKING CHECK PATH TO Models")
model_files = listdir(path_to_models)
model_files = [f for f in model_files if isfile(f) and ("lorenz" in f)]

# ---------------------------
# Work with new models directory
# ---------------------------
new_path_to_models = args.new_path_to_models
if isdir(path_to_models):
    print("NEW Directory with NEW models seems to be OK")
else:
    raise Exception("GO FUCKING CHECK NEW PATH TO NEW Models")


def reconstruct_kdtree_sort(time_series, temp):
    """


    :param time_series:

    :return: sorted_
    """
    # Time series -> z-vectors
    rec_ts = DataPreprocessor.reconstruct_lorenz(ts=time_series, template=np.array(temp))
    kdtree = cKDTree(data=templated_ts)
    zv_n_distances, zv_n_indexes = construct_distance_matrix(z_vectors=rec_ts,
                                                             kdtree=kdtree,
                                                             k_neighbors=11)
    zv_sorted, md_sorted, mi_sorted = sort_z_vectors_by_neighbors(zvector_neighbors_distances=zv_n_distances,
                                                                  zvector_neighbors_indexes=zv_n_indexes)
    return rec_ts[zv_sorted]

for model_file in model_files:
    template = parse_model_name(model_file)
    templated_ts = DataPreprocessor.reconstruct_lorenz(ts=initial_ts, template=np.array(template))
    with open(model_file, "rb") as f:
        model = load(f)
        clusters = model.clusters
        significant_clusters = model.significant_clusters
        significant_cluster_centers = np.empty(shape=(len(significant_clusters), templated_ts.shape[1]))
        for index, significant_cluster in enumerate(significant_clusters):
            significant_indexes = (clusters == significant_cluster)
            significant_cluster = templated_ts[significant_indexes].sum(axis=0)
            assert len(significant_cluster) == templated_ts.shape[1], "Dimension of cluster center is somehow wrong!"
            significant_cluster_centers[index] = significant_cluster
        model_file = PurePosixPath(model_file)
        model_file = model_file.name
        model_file = model_file + ".npy"
        model_file = PurePosixPath(new_path_to_models) / model_file
        with open(model_file.as_posix(), "wb") as ff:
            np.save(file=ff, arr=significant_cluster_centers)







# Leave only files, not directories






