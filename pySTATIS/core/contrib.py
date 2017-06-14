from __future__ import print_function
import numpy as np


def calc_factor_scores(P, D):
    """
    Calculates factor scores for each observation.
    """

    print("Factor scores for observations... ", end='')
    factor_scores = np.dot(P, np.diag(D))

    print("Done!")
    return factor_scores


def calc_partial_factor_scores(Xscaled, Q, col_indices):
    """
    Projects individual scores onto the group-level component.
    """

    print("Calculating factor scores for datasets... ", end='')

    pfs = []

    for i, val in enumerate(col_indices):
        pfs.append(np.inner(Xscaled[:, val], Q[val, :].T))

    pfs = np.array(pfs)

    print("Done!")

    return pfs


def calc_contrib_obs(fs, ev, M, D, n_obs, n_comps):
    """
    Contributions of observations (rows of the matrix)

    :return:
    """
    print("Calculating contributions of observations... ", end='.')

    contrib_obs = np.zeros([n_obs, len(D)])

    for l in range(n_comps):
        for i in range(n_obs):
            contrib_obs[i, l] = M[i, i] * fs[i, l] ** 2 / ev[l]

    print('Done!')

    return contrib_obs


def calc_contrib_var(X, Q, A, n_comps):
    """
    Contributions of variables to a group-level components.
    :return:
    """

    print("Calculating contributions of variables... ", end='')

    contrib_var = np.zeros([X.shape[1], n_comps])

    for l in range(n_comps):
        for j in range(X.shape[1]):
            contrib_var[j, l] = A[j, j] * Q[j, l] ** 2

    print('Done!')

    return contrib_var


def calc_contrib_dat(contrib_var, col_indices, n_datasets, n_comps):
    """
    Contributions of datasets/tables to the compromise.

    :return:
    """

    print("Calculating contributions of datasets... ", end='')
    contrib_dat = np.zeros([n_datasets, n_comps])

    for l in range(n_comps):
        for i, k in enumerate(col_indices):
            contrib_dat[i, l] = np.sum(contrib_var[k, l])

    print('Done!')

    return contrib_dat


def calc_partial_interia_dat(contrib_dat, ev):
    """
    Partial inertias for datasets/tables.

    :return:
    """
    print("Calculating partial inertias for the datasets... ", end='')
    pid = contrib_dat * ev

    print('Done!')

    return pid
