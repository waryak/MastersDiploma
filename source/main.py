import time
import numpy as np
from utils import reconstruct_2
from pickle import load, dump
from tqdm import tqdm
from wishart import Wishart
from runge_kutta import RungeKutta
from itertools import product

import csv


if __name__ == "__main__":


    # rk = RungeKutta(beta=8/3, rho=26, sigma=10, dt=0.1)
    # ts = rk.get_series(n_iterations=int(1e6))

    n_of_neighbors = [11]
    significance = [0.1, 0.05, 0.01, 0.005]
    combinations = list(product(n_of_neighbors, significance))

    with open("/home/vladenkov/MastersDiploma/data/ts28.npy", "rb") as f:
            reconstructed_ts = np.load(f)
            length = reconstructed_ts.shape[0]
            train_length = length // 3 * 2
            reconstructed_ts = reconstructed_ts[:train_length]
            reconstructed_ts = reconstruct_2(ts=reconstructed_ts, m=5)


    for combination in tqdm(combinations):
        k = combination[0]
        h = combination[1]
        print("Number of neighbors:", k, "and significance:", h)
        ws = Wishart(k=k, h=h)
        t1 = time.time()
        kdt = ws._fit_kd_tree(z_vectors=reconstructed_ts)
        t2 = time.time()
        m_d, m_i, v_s = ws._construct_neighbors_matrix(z_vectors=reconstructed_ts, kdtree=kdt)
        t3 = time.time()
        m_i = m_i.astype(int)
        ws._form_graph(m_d=m_d, m_i=m_i, v_s=v_s)

        path = "/home/vladenkov/MastersDiploma/data/"
        ws_name = "wishart_model_fixed" + str(k) + "_" + str(h) + ".pkl"
        path = path + ws_name
        with open(path, "wb") as f:
            dump(ws, f)
        print("Time:", t1, t2, t3)

        kdt.query()

