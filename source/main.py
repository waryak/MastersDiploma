import numpy as np
from os import makedirs, path
from runge_kutta import RungeKutta
from wishart import ParallelWishart
from utils import generate_templates, generate_arguments
from pathlib import PurePosixPath


config_dict = {"generate": False,
               "path_to_ts": "/Users/waryak/Documents/HSE/MastersDiploma/data/ts28.npy",
               "path_to_models": "/Users/waryak/Documents/HSE/MastersDiploma/data/new_models",
               "parallel": False,
               "wishart_neighbors": 11,
               "wishart_significance": 0.1,
               "n_processes": 2,
               "template_start_index": 456,
               "template_end_intex": 460,
               "template_index_step": 2,
               "template_max_distance": 10,
               "reconstruction_dimension": 5,
               "continue_pool_work": False}

TEMPLATE_MAX_DISTANCE = config_dict["template_max_distance"]
TEMPLATE_INDEX_STEP = config_dict["template_index_step"]
TEMPLATE_START_INDEX = config_dict["template_start_index"]
TEMPLATE_END_INDEX = config_dict["template_end_intex"]
PATH_TO_TS = PurePosixPath(config_dict["path_to_ts"])
PATH_TO_MODELS = PurePosixPath(config_dict["path_to_models"])
N_PROCESSES = config_dict["n_processes"]
RECONSTRUCT_DIMENSION = config_dict["reconstruction_dimension"]

if __name__ == "__main__":


    print("------------------- GENERATE OR LOAD DATA -------------------")
    if config_dict["generate"]:
        rk = RungeKutta(beta=8 / 3, rho=26, sigma=10, dt=0.1)
        ts = rk.get_series(n_iterations=int(1e6))
    else:
        assert path.exists(PATH_TO_TS.as_posix()), "--> Warning! Path with ts doesn't exist"
        with open(PATH_TO_TS.as_posix(), "rb") as f:
            ts = np.load(f)
    ts = ts[:30000]

    print("----------- CONSTREUCT DATA FOR POOL OF PROCESSES -----------")
    templates = generate_templates(n_dimension=RECONSTRUCT_DIMENSION-1,
                                   max_distance=TEMPLATE_MAX_DISTANCE,
                                   step=TEMPLATE_INDEX_STEP,
                                   start_index=TEMPLATE_START_INDEX,
                                   end_index=TEMPLATE_END_INDEX)
    if not config_dict["continue_pool_work"]:
        print("--> Running one iteration of pool workers and stopping")
        templates = templates[:N_PROCESSES]
        if templates.shape[0] < N_PROCESSES:
            print("--> Warning! Number of is less than number of processes")
            print(templates.shape, ts.shape, RECONSTRUCT_DIMENSION)
        arguments = generate_arguments(templates=templates,
                                       ts=ts,
                                       m=RECONSTRUCT_DIMENSION,
                                       n_lags=1)
        if path.exists(PATH_TO_MODELS):
            print("--> Models storage already exist. Be sure everything is under control")
        else:
            print("--> Creating folder to store models for the first time.")
            makedirs(path=PATH_TO_MODELS)
        pw = ParallelWishart(MODEL_PATH=PATH_TO_MODELS,
                             k=config_dict["wishart_neighbors"],
                             h=config_dict["wishart_significance"],
                             n_processes=N_PROCESSES)
        print("-------- RUN PARALLEL WISHART FOR ONE POOL ITERATION --------")
        result = pw.run_wishart(list_of_args=arguments)
    else:
        #Run it several times
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

