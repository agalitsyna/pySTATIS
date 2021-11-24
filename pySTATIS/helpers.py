from __future__ import print_function
from scipy import sparse
import numpy as np

def gen_affinity_input(data, type = 'cross_product', force=False):
    """
    Apply the transformation to affinity matrix in the input data. This depends on the type of STATIS used.

    :param data: The input data
    :return: The data with affinity matrix
    """

    print("Generating affinity matrices...")
    data_wa = data
    for d in data_wa:
        print(d.ID + "... ", end="")
        if d.affinity_ is None or force==True:
            if type is 'cross_product':
                d.cross_product()
            elif type is 'covariance':
                d.covariance()
            elif type is 'double_center':
                d.double_center()
            print("Done.")
        else:
            print("Affinity matrix already exists.")


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

def sparse_corrcoef(A):
    n = A.shape[1]
    rowsum = A.sum(axis=1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)
    return C.toarray()


def get_col_indices(data, ids, groups, ugroups):

    """
    Returns dict(s) that maps IDs to columns

    :return:
    """

    print('Getting indices... ', end = '')

    col_indices = []
    c = 0
    for i, u in enumerate(ids):
        col_indices.append(np.arange(c, c + data[i].n_var))
        c += data[i].n_var

    grp_indices = []
    for i, ug in enumerate(ugroups):
        ginds = []
        for g in ug:
            ginds.append(np.concatenate(list(map(col_indices.__getitem__, np.where(groups[:, i] == g)[0]))))
        grp_indices.append(ginds)

    print('Done!')

    return col_indices, grp_indices


def stack_tables(data, n_datasets, mode='sparse'):
    """
    Stacks preprocessed tables horizontally
    """

    print("Stack datasets for GSVD...", end='')

    if mode=='sparse':
        X = sparse.hstack([data[i].data_std_ for i in range(n_datasets)])
        X_scaled = sparse.hstack([data[i].data_scaled_ for i in range(n_datasets)])
    else:
        X = np.concatenate([data[i].data_std_ for i in range(n_datasets)], axis=1)
        X_scaled = np.concatenate([data[i].data_scaled_ for i in range(n_datasets)], axis=1)

    print("Done!")
    return X, X_scaled

def link_tables(data, n_datasets):
    """
    Stacks preprocessed tables horizontally
    """

    print("Linking datasets...")

    X = [data[i].data_std_ for i in range(n_datasets)]
    X_scaled = [data[i].data_scaled_ for i in range(n_datasets)]

    return X, X_scaled


def stack_linked_tables(X, mode='sparse'):
    """
    Stacks preprocessed tables horizontally
    """

    if mode=='sparse':
        X = sparse.hstack(X)
    else:
        X = np.concatenate(X, axis=1)

    return X