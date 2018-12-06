import numpy as np

import yaml
from pathlib import PurePosixPath
from os import makedirs, path, environ

from runge_kutta import RungeKutta
from wishart import ParallelWishart
from utils import generate_templates, generate_arguments

if __name__ == "__main__":

    # configs
    file_conf = environ.get('CONFIG')
    try:
        with open(file_conf, 'r') as file_conf:
            configs = yaml.load(file_conf)
    except Exception:
        raise Exception("Could not find configuration file")

    CONFIG_DATA_GENERATE = configs["data"]["generate"]
    CONFIG_DATA_PATH_TO_TS = PurePosixPath(configs["data"]["path_to_ts"])
    CONFIG_DATA_PATH_TO_MODELS = PurePosixPath(configs["data"]["path_to_models"])

    N_PROCESSES = configs["algorithm"]["parameters"]["n_processes"]
    WISHART_NEIGHBORS = configs["algorithm"]["parameters"]["wishart_neighbors"]
    assert WISHART_NEIGHBORS > 1, "Too little Wishart neighbors. One neighbor of a vector is the vector itself"
    WISHART_SIGNIFICANCE = configs["algorithm"]["parameters"]["wishart_significance"]
    RECONSTRUCT_DIMENSION = configs["algorithm"]["parameters"]["reconstruction_dimension"]

    TEMPLATE_START_INDEX = configs["algorithm"]["templates"]["template_start_index"]
    TEMPLATE_END_INDEX = configs["algorithm"]["templates"]["template_end_intex"]
    TEMPLATE_INDEX_STEP = configs["algorithm"]["templates"]["template_index_step"]
    TEMPLATE_MAX_DISTANCE = configs["algorithm"]["templates"]["template_max_distance"]



    print("------------------- GENERATE OR LOAD DATA -------------------")
    if CONFIG_DATA_GENERATE:
        rk = RungeKutta(beta=8 / 3, rho=28, sigma=10, dt=0.1)
        ts = rk.get_series(n_iterations=int(1e6))
        CONFIG_DATA_PATH_TO_TS = CONFIG_DATA_PATH_TO_TS / "ts28.npy"
        with open(CONFIG_DATA_PATH_TO_TS, "wb") as f:
            np.save(file=f, arr=ts)
    else:
        CONFIG_DATA_PATH_TO_TS = CONFIG_DATA_PATH_TO_TS / "ts28.npy"
        assert path.exists(CONFIG_DATA_PATH_TO_TS.as_posix()), "--> Warning! Path with ts doesn't exist"
        with open(CONFIG_DATA_PATH_TO_TS.as_posix(), "rb") as f:
            ts = np.load(f)
    ts = ts[:30000]

    print("----------- CONSTREUCT DATA FOR POOL OF PROCESSES -----------")
    templates = generate_templates(n_dimension=RECONSTRUCT_DIMENSION - 1,
                                   max_distance=TEMPLATE_MAX_DISTANCE,
                                   step=TEMPLATE_INDEX_STEP,
                                   start_index=TEMPLATE_START_INDEX,
                                   end_index=TEMPLATE_END_INDEX)
    # TODO: REPLACE FALSE WITH CONFIG
    if True:
        print("--> Running one iteration of pool workers and stopping")
        templates = templates[:N_PROCESSES]
        if templates.shape[0] < N_PROCESSES:
            print("--> Warning! Number of is less than number of processes")
            print(templates.shape, ts.shape, RECONSTRUCT_DIMENSION)
        arguments = generate_arguments(templates=templates,
                                       ts=ts,
                                       m=RECONSTRUCT_DIMENSION,
                                       n_lags=1)
        if path.exists(CONFIG_DATA_PATH_TO_MODELS):
            print("--> Models storage already exist. Be sure everything is under control")
        else:
            print("--> Creating folder to store models for the first time.")
            makedirs(path=CONFIG_DATA_PATH_TO_MODELS)
        pw = ParallelWishart(MODEL_PATH=CONFIG_DATA_PATH_TO_MODELS,
                             k=WISHART_NEIGHBORS,
                             h=WISHART_SIGNIFICANCE,
                             n_processes=N_PROCESSES)
        print("-------- RUN PARALLEL WISHART FOR ONE POOL ITERATION --------")
        result = pw.run_wishart(list_of_args=arguments)
    else:
        # Run it several times
        pass

    #
    # with open(PurePosixPath(config_dict), "rb") as f:
    #         reconstructed_ts = np.load(f)
    #         length = reconstructed_ts.shape[0]
    #         train_length = length // 3 * 2
    #         reconstructed_ts = reconstructed_ts[:train_length]
    #         reconstructed_ts = reconstruct_2(ts=reconstructed_ts, m=5)

    #
    # for combination in tqdm(combinations):
    #     k = combination[0]
    #     h = combination[1]
    #     print("Number of neighbors:", k, "and significance:", h)
    #     ws = Wishart(k=k, h=h)
    #     kdt = ws._fit_kd_tree(z_vectors=reconstructed_ts)
    #     m_d, m_i, v_s = ws._construct_neighbors_matrix(z_vectors=reconstructed_ts, kdtree=kdt)
    #     m_i = m_i.astype(int)
    #     ws._form_graph(m_d=m_d, m_i=m_i, v_s=v_s)
    #
    #     path = "/home/vladenkov/MastersDiploma/data/"
    #     ws_name = "wishart_model_fixed" + str(k) + "_" + str(h) + ".pkl"
    #     path = path + ws_name
    #     with open(path, "wb") as f:
    #         dump(ws, f)
    #
    #     kdt.query()

    # n_of_neighbors = [11]
    # significance = [0.1, 0.05, 0.01, 0.005]
    # combinations = list(product(n_of_neighbors, significance))
