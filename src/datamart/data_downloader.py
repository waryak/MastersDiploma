import numpy as np
from os import makedirs, path
from pathlib import PurePosixPath
from src.algo.runge_kutta import RungeKutta


class DataDownloader:
    """
    Class to prepare data structure, directories and download data itself
    """

    def __init__(self, mount_point_path):
        self.path_to_data = None  # It is declared later when data is downloaded or generated
        self.PATH_TO_TS = PurePosixPath(mount_point_path) / "data_ts"
        self.PATH_TO_MODELS = PurePosixPath(mount_point_path) / "data_clusters"

        if not path.exists(self.PATH_TO_TS.__str__()):
            print("-> Creating directories for time series's! (ATTENTION HERE)")
            makedirs(self.PATH_TO_TS.__str__())
        if not path.exists(self.PATH_TO_MODELS.__str__()):
            print("-> Creating directories for models! (ATTENTION HERE)")
            makedirs(self.PATH_TO_MODELS.__str__())

    def _download_lorenz(self):
        """
        Generate modeled lorenz time series
        :return:
        """
        pass

    def generate_lorenz(self, beta, rho, sigma, dt, size):
        """
        Generating Lorenz modeled time series
        :param beta:
        :param rho:
        :param sigma:
        :param dt:
        :return:
        """

        file_name = "lorenz_{rho}_{sigma}_{dt}_{size}.npy".format(rho=rho, sigma=sigma, dt=str(dt), size=size)
        path_to_file = PurePosixPath(self.PATH_TO_TS) / file_name
        self.path_to_data = path_to_file
        if path.exists(path_to_file):
            print("-> Time series with parameters rho={rho}, sigma={sigma}, dt={dt}, size={size} already exists". \
                  format(rho=rho, sigma=sigma, dt=str(dt), size=size))
        else:
            print("-> Creating time series with parameters rho={rho}, sigma={sigma}, dt={dt}, size={size}". \
                  format(rho=rho, sigma=sigma, dt=str(dt), size=size))
            rk = RungeKutta(beta=beta, rho=rho, sigma=sigma, dt=dt)
            ts = rk.get_series(n_iterations=size)
            with open(path_to_file.as_posix(), "wb") as f:
                np.save(file=f, arr=ts)

    def get_path_to_data(self):
        """

        :return:
        """
        return self.path_to_data

    def get_data(self):
        """

        :return:
        """
        path_to_data = self.get_path_to_data()
        with open(path_to_data.as_posix(), "rb") as f:
            data = np.load(file=f)
        return data







