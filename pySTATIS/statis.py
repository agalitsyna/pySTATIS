'''
Python implementation of STATIS
'''

from __future__ import print_function

import numpy as np
from scipy.stats.mstats import zscore

from core.helpers import gen_affinity_input, get_ids, get_groups
from core.decomposition import rv_pca, get_A, get_M, stack_tables, gsvd, get_col_indices
from core.contrib import calc_factor_scores, calc_partial_factor_scores, calc_contrib_var, calc_contrib_obs, \
    calc_contrib_dat, calc_partial_interia_dat


class STATISData(object):
    def __init__(self, X, ID, ev=None, groups=['group_1', 'group_2'], normalize=('zscore', 'norm_one'), col_names=None,
                 row_names=None):
        """
        X: input variables for a single entity
        ID: ID of the entity; can be a set
        ev: eigenvalues of the X columns, in case that X are principal components
        col_names, row_names: labels for rows and columns
        normalize: normalization method to use (None, 'zscore', 'double_center')
        """

        self.data = X
        self.ev = ev
        self.ID = ID
        self.n_var = X.shape[1]
        self.groups = groups
        self.data_std_ = None
        self.affinity_ = None

        if col_names is None:
            self.col_names = ['col_%s' % str(i).zfill(5) for i in range(X.shape[1])]
        else:
            self.col_names = col_names

        if row_names is None:
            self.row_names = ['row_%s' % str(i).zfill(5) for i in range(X.shape[0])]
        else:
            self.row_names = col_names

        self.normalize(method=normalize)

        if ev is not None:
            self.data_scaled_ = self.data_std_ * self.ev
        else:
            self.data_scaled_ = self.data_std_

    def normalize(self, method=None):

        if method is 'None':
            print("Not performing any normalizations")
            self.data_std_ = self.data
        else:
            temp = self.data
            for m in method:
                if m is 'zscore':
                    temp = zscore(temp, axis=0, ddof=1)

                elif m is 'norm_one':
                    temp = temp / np.linalg.norm(temp)

                else:
                    print("Method not implemented, use 'zscore' and/or 'double_center'")

                self.data_std_ = temp

    def cross_product(self):

        self.affinity_ = self.data_std_.dot(self.data_std_.T)

    def covariance(self):

        self.affinity_ = np.cov(self.data_std_, rowvar=False)

    def double_center(self):

        assert self.data_std_.shape[0] == self.data_std_.shape[
            1], "Error: double centering requires a square matrix (correlation or covariance)"

        cmat = np.eye(self.data_std_.shape[0]) - np.ones_like(self.data_std_) * data_std_.shape[0]
        self.affinity_ = 0.5 * cmat.dot(self.data_std_.dot(cmat))


class STATIS(object):
    def __init__(self):

        self.data = None
        self.n_datasets = None
        self.n_observations = None
        self.inds_ = None

        self.A_ = None
        self.M_ = None
        self.X_ = None

        self.P_ = None  # Left singular vectors
        self.Q_ = None  # Right singular vectors
        self.D_ = None  # Singular values

        self.n_comps_ = None

        self.groups_ = None
        self.ugroups_ = None
        self.n_groupings_ = None

        self.factor_scores_ = None
        self.partial_factor_scores_ = None
        self.col_indices_ = None

        self.contrib_obs_ = None
        self.contrib_var_ = None
        self.contrib_dat_ = None
        self.partial_inertia_dat_ = None
        self.contrib_grp_ = None

    def fit(self, data):

        """
        Main method to run STATIS on input example_data. Input example_data must be a list of STATISData objects.
        
        :param data: List of STATISdata objects
        """

        import time

        t0 = time.time()
        self.n_datasets = len(data)
        self.n_observations = data[0].data.shape[0]

        # Pre-process
        self.data = gen_affinity_input(data)
        self.ids_ = get_ids(self.data)
        self.groups_, self.ugroups_, self.n_groupings_ = get_groups(self.data)

        # Get weights and prepare for GSVD
        self.table_weights_ = rv_pca(self.data, self.n_datasets)
        self.A_ = get_A(self.data, self.table_weights_, self.n_datasets)
        self.M_ = get_M(self.n_observations)
        self.X_, self.X_scaled_ = stack_tables(self.data, self.n_datasets)

        # Decompose
        self.P_, self.D_, self.Q_, self.ev_, self.n_comps_ = gsvd(self.X_, self.M_, self.A_)

        # Get indices for each dataset
        self.col_indices_, self.grp_indices_ = get_col_indices(self.data, self.ids_, self.groups_, self.ugroups_)

        # Post-process
        self.factor_scores_ = calc_factor_scores(self.P_, self.D_)
        self.partial_factor_scores_ = calc_partial_factor_scores(self.X_scaled_, self.Q_, self.col_indices_)
        self.contrib_obs_ = calc_contrib_obs(self.factor_scores_, self.ev_, self.M_, self.D_, self.n_observations,
                                             self.n_comps_)
        self.contrib_var_ = calc_contrib_var(self.X_, self.Q_, self.A_, self.n_comps_)
        self.contrib_dat_ = calc_contrib_dat(self.contrib_var_, self.col_indices_, self.n_datasets, self.n_comps_)
        self.partial_inertia_dat_ = calc_partial_interia_dat(self.contrib_dat_, self.ev_)

        print('STATIS finished successfully in %.3f seconds' % (time.time() - t0))


class dualSTATIS(STATIS):
    def gen_affinity_input(self):
        for d in self.data:
            d.covariance()


class COVSTATIS(STATIS):
    def gen_affinity_input(self):
        for d in self.data:
            d.double_center()
