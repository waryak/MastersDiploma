import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import distance


class Lyapunov:
    """
    Class which emplements resenshein algorithm
    """

    def __init__(self, i, m, J, k):
        """

        """
        self.i = i
        self.m = m
        self.J = J
        self.k = k

    def _reconstruct(self, ts):
        """
        """
        ts_length = ts.shape[0]
        n_samples = ts_length // (self.m + self.J)
        ts = ts[:n_samples * (self.m + self.J)]
        ts_reconstructed = ts.reshape((n_samples, self.m + self.J))
        if self.J > 0:
            ts_reconstructed = ts_reconstructed[:, :-self.J]
        return ts_reconstructed

    def _fit_kd_tree(self, data):
        kdt = cKDTree(data=data)
        return kdt

    def _generator(self, data, kdtree):
        data_length = len(data)
        for index in range(data_length - 1):
            element = data[index]
            distances, neighbor_index = kdtree.query(x=element, k=self.k)
            neighbor_index = neighbor_index[self.k - 1]
            distance_1 = distances[self.k - 1]
            if neighbor_index < (data_length - 1):
                pair_j_1 = data[index + 1]
                pair_j_2 = data[neighbor_index + 1]
                distance_2 = distance.euclidean(u=pair_j_1, v=pair_j_2)
                if (distance_2 == 0) | (distance_1 == 0) | np.isnan(distance_1) | np.isnan(distance_2):
                    continue
                yield np.log(distance_2 / distance_1)
            else:
                continue

    def compute_lyapunov_exponent(self):
        pass



