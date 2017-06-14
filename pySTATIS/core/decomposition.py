from __future__ import print_function

def rv_pca(data, n_datasets):
    """
    Get weights for tables by calculating their similarity.

    :return:
    """

    from scipy.sparse.linalg import eigs
    import numpy as np

    print("Rv-PCA")
    C = np.zeros([n_datasets, n_datasets])

    print("Rv-PCA: Hilbert-Schmidt inner products... ", end='')
    for i in range(n_datasets):
        for j in range(n_datasets):
            C[i, j] = np.sum(data[i].affinity_ * data[j].affinity_)

    print('Done!')

    print("Rv-PCA: Decomposing the inner product matrix... ", end ='')
    _, u = eigs(C, k=1)
    print("Done!")
    weights = np.real(u) / np.sum(np.real(u))

    print("Rv-PCA: Done!")

    return weights

def get_A(data, table_weights, n_datasets):
    """
    Dataset masses.

    """
    import numpy as np

    print("Dataset masses... ", end='')
    a = np.concatenate([np.repeat(table_weights[i], data[i].n_var) for i in range(n_datasets)])
    print("Done!")
    return np.diag(a)

def get_M(n_obs):
    """
    Masses for observations. These are assumed to be equal.
    """

    import numpy as np

    print("Observation masses: Done!")

    return np.eye(n_obs) / n_obs

def stack_tables(data, n_datasets):
    """
    Stacks preprocessed tables horizontally
    """

    import numpy as np

    print("Stack datasets for GSVD...", end='')

    X = np.concatenate([data[i].data_std_ for i in range(n_datasets)], axis=1)
    X_scaled = np.concatenate([data[i].data_scaled_ for i in range(n_datasets)], axis=1)

    print("Done!")
    return X, X_scaled

def get_col_indices(data, ids, groups, ugroups):

    """
    Returns dict(s) that maps IDs to columns

    :return:
    """

    print('Getting indices... ', end = '')

    import numpy as np

    col_indices = []
    c = 0
    for i, u in enumerate(ids):
        col_indices.append(np.arange(c, c + data[i].n_var))
        c += data[i].n_var

    grp_indices = []
    for i, ug in enumerate(ugroups):
        ginds = []
        for g in ug:
            ginds.append(np.concatenate(map(col_indices.__getitem__, np.where(groups[:, i] == g)[0])))
        grp_indices.append(ginds)

    print('Done!')

    return col_indices, grp_indices

def gsvd(X, M, A):
    """
    Generalized SVD

    :param X:
    :param M:
    :param A:
    :return:
    """
    import numpy as np

    print("GSVD")

    print("GSVD: Weights... ", end='')
    Xw = np.dot(np.sqrt(M), np.dot(X, np.sqrt(A)))
    print("Done!")

    print("GSVD: SVD... ", end='')
    [P_, D, Q_] = np.linalg.svd(Xw, full_matrices=False)
    print('Done!')

    print("GSVD: Factor scores and eigenvalues... ", end='')
    Mp = np.power(np.diag(M), -0.5)
    Ap = np.power(np.diag(A), -0.5)

    P = np.dot(np.diag(Mp), P_)
    Q = np.dot(np.diag(Ap), Q_.T)
    ev = np.power(D, 2)

    n_comps = len(D)
    print('Done!')

    return P, D, Q, ev, n_comps