import numpy as np
from src.network.messages.utils import int2base, digs


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
        self.current_template = 0

    def _template_from_str_to_int(self, template):
        """
        String representation of template to integer representation of templete
        :Str, List[str] template: List with strings, which represent distance between point.
                        e.g. ["0", "a", "9", "b"] will result in [0, 10, 9, 11]
        :return: List[int] template
        """
        if type(template) is list:
            template = [digs.find(element) for element in template]
        elif type(template) is str:
            template = template.rjust(self.template_size, "0")
            template = list(template)
            template = [digs.find(element) for element in template]
        else:
            raise Exception("-> Template of wrong type %s" % type(template))
        return template

    def _template_from_int_to_str(self, template):
        """
        Converts the int or list of int's to a list of strings
        :param template: int / list[int]
        :return: list[str]
        """
        if type(template) is int:
            template = int2base(x=template, base=self.max_template_distance + 1)
            template = template.rjust(self.template_size, "0")
            template = list(template)
        elif type(template) is list:
            template = [int2base(element) for element in template]
        return template

    def _check_templates_validity(self, template):
        """
        Check if templates satisfies the borders for the template distances.
        :List[int] template: template with integers and template distances
        :return: True/False
        """

        def check_element(x):
            return (x >= self.min_template_distance) & (x <= self.max_template_distance)

        checked_template = list(map(check_element, template))
        if all(checked_template):
            return True
        else:
            # print("Element doesn't fit diapason %i-%i" % (self.min_template_distance, self.max_template_distance))
            return False

    def next_planned_template(self, method="concurrent", step=5):
        """

        :strr method: object
        :return:
        """
        assert (type(step) is int) & (step >= 1), "Template step is not integer or not big enough "
        # TODO: Repeat while the template satisfies the conditions
        next_template = self._template_from_int_to_str(self.current_template)
        next_template = self._template_from_str_to_int(next_template)
        # Skip those templates, which does not satisfies the specification.
        while not self._check_templates_validity(template=next_template):

            if method == "random":
                next_template = np.random.random_integers(low=self.min_template_distance,
                                                          high=self.max_template_distance,
                                                          size=self.template_size - 1)
            elif method == "concurrent":
                self.current_template = self.current_template + step
                if self.current_template > self.max_template_distance ** self.template_size:
                    print("-> Run out of templates")
                    return False
                next_template = self._template_from_int_to_str(self.current_template)
                next_template = self._template_from_str_to_int(next_template)
                # print("--> ", next_template, self._check_templates_validity(template=next_template))
            else:
                raise Exception("-> Unspecified method to generate templates. Must be \"random\" or \"concurrent\"")
        self.current_template = self.current_template + step
        return next_template

    def next_planned_significance(self):
        """

        :return:
        """
        return 0.1

    def next_planned_neighbors(self):
        """
        tm = TemplateManager(5, 10, 0)
        :return:
        """
        return 11

