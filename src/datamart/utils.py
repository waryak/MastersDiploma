
def parse_model_name(model_name: str):
    """
    Parse the name of the model file and extracts the template distances from it
    :param model_name: filename; e.g.: lorenz_5_2_5_6.pkl
    :return: list with template distances
    """
    model_name = model_name.split("_")
    # Get rid of the name
    model_name = model_name[1:]
    # Get rid of last two parameters
    model_name = model_name[:-2]
    template_distances = list(map(int, model_name))
    return template_distances


