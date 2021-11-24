from __future__ import print_function
import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse
from sklearn.utils.extmath import randomized_svd


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

def get_A_STATIS(data, table_weights, n_datasets, mode='sparse'):
    """
    Dataset masses.
    """

    print("Dataset/variable masses... ", end='')

    a = np.concatenate([np.repeat(table_weights[i], data[i].n_var) for i in range(n_datasets)])

    if mode=='sparse':
        return sparse.diags(a, format='csr')
    else:
        return np.diag(a)

def get_A_ANISOSTATIS(table_weights):

    return np.diag(table_weights)


def get_M(n_obs, mode='sparse'):
    """
    Masses for observations. These are assumed to be equal.
    """

    if mode=='sparse':
        masses = sparse.eye(n_obs, format='csr') / n_obs
    else:
        masses = np.eye(n_obs) / n_obs

    print("Observation masses: Done!")
    return(masses)


def gsvd(X, M, A, n_comps = 10, mode='sparse'):
    """
    Generalized SVD
    """

    print("GSVD")
    print("GSVD: Weights... ", end='')
    Xw = np.dot(np.sqrt(M), np.dot(X, np.sqrt(A)))
    print("Done!")

    print("GSVD: SVD... ", end='')
    [P_, D, Q_] = randomized_svd(Xw, n_comps)
    # print(P_, D, Q_)

    #P_ = P_[:,0:n_comps]
    #D = D[0:n_comps]
    #Q_ = Q_[0:n_comps,:]
    print('Done!')

    print("GSVD: Factor scores and eigenvalues... ", end='')
    if mode=='sparse':
        Mp = np.power(M.diagonal(0), -0.5)
        Ap = np.power(A.diagonal(0), -0.5)
    else:
        Mp = np.power(np.diag(M), -0.5)
        Ap = np.power(np.diag(A), -0.5)

    P = np.dot(np.diag(Mp), P_)
    Q = np.dot(np.diag(Ap), Q_.T)
    ev = np.power(D, 2)

    print('Done!')

    return P, D, Q, ev