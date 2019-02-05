import numpy as np

class TemplateManager:

    def __init__(self, template_size, max_template_distance, min_template_distance):
        """

        :param max_template_distance: Max number of points between two nearest points in a template
        :param min_template_distance: Min number of points between two nearest points in a template
        :param template_size: Number of points in a template <-> sizeof the z-vector
        """
        self.template_size = template_size
        self.max_template_distance = max_template_distance
        self.min_template_distance = min_template_distance

    def next_planned_template(self):
        """

        :return:
        """
        next_template = np.random.random_integers(low=self.min_template_distance,
                                                  high=self.max_template_distance,
                                                  size=self.template_size - 1)
        return next_template

    def next_planned_significance(self):
        """

        :return:
        """
        return 0.1

    def next_planned_neighbors(self):
        """

        :return:
        """
        return 11



