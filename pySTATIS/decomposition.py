from __future__ import print_function
import numpy as np
from scipy.sparse.linalg import eigs

def rv_pca(data, n_datasets):
    """
    Get weights for tables by calculating their similarity.

    :return:
    """

    print("Rv-PCA")
    C = np.zeros([n_datasets, n_datasets])

    print("Rv-PCA: Hilbert-Schmidt inner products... ", end='')
    for i in range(n_datasets):
        for j in range(n_datasets):
            C[i, j] = np.sum(data[i].affinity_ * data[j].affinity_)

    print('Done!')

    print("Rv-PCA: Decomposing the inner product matrix... ", end ='')
    e, u = eigs(C, k=8)
    print("Done!")
    weights = (np.real(u[:,0]) / np.sum(np.real(u[:,0]))).flatten()

    print("Rv-PCA: Done!")

    return weights, np.real(e), np.real(u)

def aniso_c1(X, M):
    """
    Calculate weights using ANISOSTATIS criterion 1
    :param X:
    :param M:
    :return: New weights
    """
    print("Computing ANISOSTATIS Criterion 1 weights...", end='')

    C = np.power(X.T.dot(M.dot(X)), 2)
    ev, Mn = eigs(C, k=1)
    Mn = np.real(Mn)
    Mn = np.concatenate(Mn / np.sum(Mn))

    print('Done!')
    return Mn, ev

def get_A(data, table_weights, n_datasets, flavor):
    """
    Dataset masses.

    """

    print("Dataset/variable masses... ", end='')
    if flavor is 'STATIS':
        a = np.concatenate([np.repeat(table_weights[i], data[i].n_var) for i in range(n_datasets)])
    elif 'ANISOSTATIS' in flavor:
        a = table_weights
    else:
        raise ValueError('You specified a non-existing STATIS flavor!')
    print("Done!")

    return np.diag(a)

def get_M(n_obs):
    """
    Masses for observations. These are assumed to be equal.
    """

    print("Observation masses: Done!")

    return np.eye(n_obs) / n_obs

def stack_tables(data, n_datasets):
    """
    Stacks preprocessed tables horizontally
    """

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

def gsvd(X, M, A, n_comps = 10):
    """
    Generalized SVD

    :param X:
    :param M:
    :param A:
    :return:
    """

    print("GSVD")
    print("GSVD: Weights... ", end='')
    Xw = np.dot(np.sqrt(M), np.dot(X, np.sqrt(A)))
    print("Done!")

    print("GSVD: SVD... ", end='')
    [P_, D, Q_] = np.linalg.svd(Xw, full_matrices=False)

    P_ = P_[:,0:n_comps]
    D = D[0:n_comps]
    Q_ = Q_[0:n_comps,:]
    print('Done!')

    print("GSVD: Factor scores and eigenvalues... ", end='')
    Mp = np.power(np.diag(M), -0.5)
    Ap = np.power(np.diag(A), -0.5)

    P = np.dot(np.diag(Mp), P_)
    Q = np.dot(np.diag(Ap), Q_.T)
    ev = np.power(D, 2)

    print('Done!')

    return P, D, Q, ev