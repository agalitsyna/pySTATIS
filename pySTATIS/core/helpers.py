def gen_affinity_input(data):
    """
    Apply the transformation to affinity matrix in the input data. This depends on the type of STATIS used.

    :param data: The input data
    :return: The data with affinity matrix
    """

    data_wa = data
    for d in data_wa:
        d.cross_product()

    return data_wa


def get_groups(data):
    """
    Extracts the information about groups membership from the datasets.

    :return: Group memeberships, unique groups for each level, number of grouping variables
    """

    import numpy as np

    groups = []
    for g in data:
        groups.append(g.groups)
    groups = np.array(groups)

    ugroups = []
    for i in range(groups.shape[1]):
        ugroups.append(np.unique(groups[:, i]))

    n_groupings = len(ugroups)

    return groups, ugroups, n_groupings


def get_ids(data):
    """
    Extracts dataset IDs.

    :param data: The input data
    :return: IDs
    """
    ids = []

    for i in data:
        ids.append(i.ID)

    return ids
