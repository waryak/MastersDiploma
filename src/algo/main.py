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
    ds = DataSaver(template.astype(str), wishart_neighbors, wishart_significance,
                   local_path=CONFIG_MOUNT_POINT,
                   base_name="lorenz")

    print("------------------- GENERATE OR LOAD DATA -------------------")
    dd.generate_lorenz(beta=8 / 3, rho=28, sigma=10, dt=0.1, size=int(3e6))

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





    #
    # print("----------- CONSTRUCT DATA FOR POOL OF PROCESSES ------------")
    # templates = generate_templates(n_dimension=RECONSTRUCT_DIMENSION - 1,
    #                                max_distance=TEMPLATE_MAX_DISTANCE,
    #                                step=TEMPLATE_INDEX_STEP,
    #                                start_index=TEMPLATE_START_INDEX,
    #                                end_index=TEMPLATE_END_INDEX)
    # # TODO: REPLACE FALSE WITH CONFIG
    # if True:
    #     print("--> Running one iteration of pool workers and stopping")
    #     templates = templates[:N_PROCESSES]
    #     if templates.shape[0] < N_PROCESSES:
    #         print("--> Warning! Number of is less than number of processes")
    #         print(templates.shape, ts.shape, RECONSTRUCT_DIMENSION)
    #     arguments = generate_arguments(templates=templates,
    #                                    ts=ts,
    #                                    m=RECONSTRUCT_DIMENSION,
    #                                    n_lags=1)
    #     if path.exists(CONFIG_DATA_PATH_TO_MODELS):
    #         print("--> Models storage already exist. Be sure everything is under control")
    #     else:
    #         print("--> Creating folder to store models for the first time.")
    #         makedirs(CONFIG_DATA_PATH_TO_MODELS.as_posix())
    #     pw = ParallelWishart(MODEL_PATH=CONFIG_DATA_PATH_TO_MODELS,
    #                          k=WISHART_NEIGHBORS,
    #                          h=WISHART_SIGNIFICANCE,
    #                          n_processes=N_PROCESSES)
    #     print("-------- RUN PARALLEL WISHART FOR ONE POOL ITERATION --------")
    #     result = pw.run_wishart(list_of_args=arguments)
    # else:
    #     # Run it several times
    #     pass


if __name__ == "__main__":

    t = np.array([1, 2, 3, 4])
    wishart_neighbors = 11
    wishart_significance = 0.2
    main(template=t,
         wishart_neighbors=wishart_neighbors,
         wishart_significance=wishart_significance)
