import yaml
import numpy as np
from os import environ
from pathlib import PurePosixPath

from src.algo.wishart import Wishart
from src.datamart.data_saver import DataSaver
from src.datamart.data_downloader import DataDownloader
from src.datamart.data_preprocesser import DataPreprocessor


def main(template, wishart_neighbors, wishart_significance):
    """
    Main function to perform configs loading and wishart calculations
    :return:
    """

    print("------------------ LOADING CONFIGURATIONS -------------------")
    file_conf = environ.get('CONFIG')
    try:
        with open(file_conf, 'r') as file_conf:
            configs = yaml.load(file_conf)
    except Exception:
        raise Exception("Could not find configuration file")

    CONFIG_MOUNT_POINT = PurePosixPath(configs["data"]["mount_point"])

    dd = DataDownloader(mount_point_path=CONFIG_MOUNT_POINT)
    dp = DataPreprocessor(dd=dd, kind_of_data="lorenz", train_spec=None)
    ds = DataSaver(local_path=CONFIG_MOUNT_POINT,
                   base_name="lorenz",
                   template=template,
                   **{"wishart_neighbors": wishart_neighbors,
                      "wishart_significance": wishart_significance})

    print("------------------- GENERATE OR LOAD DATA -------------------")
    dd.generate_lorenz(beta=8 / 3, rho=28, sigma=10, dt=0.1, size=int(3e5))

    print("---------------------- PREPROCESS DATA ----------------------")
    reconstructed_ts = dp.prepare_data(template=template)

    print("------------------------ RUN WISHART ------------------------")
    print("-> Using parameters: \n      template: {template}\n"
          "      wishart_neighbors: {wishart_neighbors}\n      wishart_significance: {wishart_significance} ". \
          format(template="-".join(template.astype(str)),
                 wishart_neighbors=wishart_neighbors,
                 wishart_significance=wishart_significance))
    ws = Wishart(wishart_neighbors=wishart_neighbors,
                 significance_level=wishart_significance)

    print("--> Fitting KDTree")
    kdt = ws._fit_kd_tree(z_vectors=reconstructed_ts)
    print("--> Constructing vertex data")
    m_d, m_i, v_s = ws._construct_neighbors_matrix(z_vectors=reconstructed_ts, kdtree=kdt)
    m_i = m_i.astype(int)
    print("--> Fitting the graph")
    ws._form_graph(m_d=m_d, m_i=m_i, v_s=v_s)
    print("--> Finding the cluster centers")
    ws._form_cluster_centers(data=m_d)
    print("--> Fitting clusters KDTree")
    ws._cluster_kdtree(n_lags=len(template))

    print("------------------- SAVING/UPLOADING DATA -------------------")
    ds.save_to_volume(ws)

    return len(ws.clusters_completenes), len(ws.significant_clusters)



# if __name__ == "__main__":
#
#     t = np.array([1, 2, 3, 4])
#     wishart_neighbors = 11
#     wishart_significance = 0.2
#     main(template=t,
#          wishart_neighbors=wishart_neighbors,
#          wishart_significance=wishart_significance)
