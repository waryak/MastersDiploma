import pandas as pd
from src.algo


class DataManager:
    """
    Class to prepare data structure, directories and download data itself
    """
    def __init__(self, PATH_TO_TS, PATH_TO_MODELS):
        self.PATH_TO_TS = PATH_TO_TS
        self.PATH_TO_MODELS = PATH_TO_MODELS

    def _download_generate_lorenz(self):
        """
        Generate modeled lorenz time series
        :return:
        """

