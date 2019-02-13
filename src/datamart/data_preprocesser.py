import numpy as np

from src.datamart.data_downloader import DataDownloader


class DataPreprocessor:
    """

    """
    def __init__(self, dd: DataDownloader, kind_of_data, train_spec):
        """

        :param kind_of_data: Determines kind of data: <Lorenz> or <Financial>
        """
        self.kind_of_data = kind_of_data
        self.train_spec = train_spec
        self.data_downloader = dd

    @staticmethod
    def _reconstruct_lorenz(ts: np.ndarray, template: np.ndarray):
        """
        Reconstructing the time series to array of z-vectors. For example:
        Time series: [1,2,3,4,5,6,7,8,9,10,11,12]
        Template: [1,1,5,2]
        Result: [[1, 2, 3, 8, 10],
                 [2, 3, 4, 9, 11],
                 [3, 4, 5, 10, 12]]

        :param ts: Original time series to reconstruct
        :param template: Template to reconstruct the ts
        :return: Reconstructed time series
        """

        ts_list = [ts[:-np.sum(template)].reshape(-1, 1)]

        for offset in template.cumsum()[:-1]:
            offset_ts = ts[offset:-(template.sum() - offset)].reshape(-1, 1)
            ts_list.append(offset_ts)
        ts_list.append(ts[np.sum(template):].reshape(-1, 1))
        reconstructed_ts = np.concatenate(ts_list, axis=1)
        return reconstructed_ts


    def preprocess_lorenz(self):
        """
        Fynction to preprocess the modeled lorenz time-series.
        For the lorenz case it does:
        1. The scaling/norming of the time-series
        2.
        :return:
        """
        ts = self.data_downloader.get_data()

        pass

    def preprocess_financial(self):
        """

        :return:
        """
        pass


    def prepare_data(self, template):
        """
        Preprocessing all needed data
        :return:
        """
        ts = self.data_downloader.get_data()
        if self.kind_of_data == "lorenz":
            print("-> Preprocessing modeles lorenz data")
            return self._reconstruct_lorenz(ts=ts,
                                           template=template)
        elif self.kind_of_data == "financial":
            print("-> Preprocessing real financial data")
            return
        else:
            raise Exception("Unrecognized type of data. Should be \"lorenz\"/\"financial\"")

