from os import makedirs, path
from pickle import dump
from pathlib import PurePosixPath


class DataSaver:
    """
    Class that implements methods to save models to volumes and upload them to database
    """

    def __init__(self, local_path, base_name, template, **kwargs):
        """

        :param local_path:
        :param base_name:
        :param template:
        :param kwargs:
        """
        print("DEBUG", kwargs)
        print(template.astype(str))
        self.LOCAL_PATH = PurePosixPath(local_path)
        self.model_name = base_name + "_" + "_".join(template.astype(str))
        self.model_name = self.model_name + "_" + "_".join(list(map(str, kwargs.values())))
        self.local_path_to_model = self.LOCAL_PATH / self.model_name


    # TODO: Hash functions for model's names to distinguish them

    def _check_paths(self):
        """
        Checks local paths and tries to create them if they don't exist.
        Throws exception if failed.
        :return: True if path exists
        """
        if path.exists(self.LOCAL_PATH):
            return None
        else:
            print("-> Creating local mounted path inside the container. Pay attention!")
            try:
                makedirs(self.LOCAL_PATH.as_posix())
            except Exception:
                raise Exception("Could not create mounted path")

    def _checks_connections(self):
        """
        Checks
        :return:
        """
        return None

    def save_to_volume(self, obj):
        """
        Save to local disk to docker-volume
        :return:
        """
        self._check_paths()

        path_to_model = self.LOCAL_PATH / self.model_name
        with open(path_to_model, "wb") as f:
            dump(obj, f)

    def _load_to_database(self):
        """
        Loads data to database
        :return:
        """
        pass
